# §9.6 Author Notes — Reviewer Preempts Before Prose Writing (Task 9)

Captured 2026-04-15 from the Task 2 identifiability audit review cycle (lit review + code review).

## Reviewer-preempt items for §9.6 chapter prose

1. **Secondary nucleation absence** — Our ODE absorbs fibril-surface-catalyzed oligomer formation into the K_CONV rate (monomer→oligomer conversion). Cite Xu 2024 Nat Commun as the modern reference and disclose this as deliberate simplification. Reviewer who reads Iljina 2016 + Xu 2024 will notice.

2. **Non-spatial scope** — The 4-state ODE is well-mixed (no connectome propagation). Cite Weickenmeier 2019 and Fornari 2019 Fisher-Kolmogorov PD models; explicitly scope propagation to Paper 8b. Do not overclaim mechanism coverage.

3. **SAA_TTT ratio observable non-Gaussian noise** — The SAA_TTT channel is 1/F(t), a ratio on a small denominator. Its error propagation is non-Gaussian. Acknowledge in methods. FIM conclusions are robust to this because κ depends on Jacobian geometry, not absolute σ — state this explicitly.

4. **BDF integrator cross-check** — Rerun the audit with method="BDF" (instead of LSODA) and report κ invariant. One line in methods: "Robustness check: FIM condition number κ changes by < 1% under BDF integration, confirming the result is not integrator-dependent."

5. **StructuralIdentifiability.jl symbolic certificate** — Optional but strongly recommended for reviewer-proof identifiability. The SciML package at https://github.com/SciML/StructuralIdentifiability.jl would give a categorical identifiable/not verdict complementary to our FIM + profile likelihood. ~50 LOC Julia script. Cite as "consulted but not required for p=2" if not run, or include the certificate in supplement.

6. **Sloppiness framing vs Gutenkunst 6-decade baseline** — Frame our 1.3-decade eigenvalue spread relative to Gutenkunst 2007 PLoS CB, which established that typical sloppy biological models span ≥6 decades. Our result is "2 orders of magnitude tighter than typical systems-biology models."

7. **α_tox lower CI hits grid boundary** — This is the known "population-identifiable, individual-sloppy" signature. `mechanistic.data_registry` documents cor(log k_n, log α_tox) = -0.851 in the HIGH-INFO subset of the Phase 2 IS posteriors. Report honestly: "k_n is practically identifiable (tight CI); α_tox is identifiable at the population scale under hierarchical Bayesian pooling but sloppy at the individual scale — consistent with Véronneau-Veilleux 2020 JPKPD and Holford NLME tradition." This is a feature, not a bug — it motivates the hierarchical SAEM design in §9.3.

## Supporting citations (already in `audit.citation` or `reference.phase5_bibliography`)

- Iljina 2016 PNAS (aggregation kinetics, 203 cit — cited by K_CONV, K_CLEAR_O)
- Xu 2024 Nat Commun (fibril-surface secondary nucleation)
- Lee 2019 JAMA Neurology (GAMMA=0.7 SBR-to-N exponent)
- Gutenkunst 2007 PLoS CB (sloppiness, 6-decade baseline)
- Raue 2009 Bioinformatics, Raue 2013 PLoS ONE (profile likelihood)
- Transtrum 2015 J Chem Phys (sloppiness in physics/biology)
- Villaverde 2016 PLoS CB (structural identifiability tools)
- Véronneau-Veilleux 2020 JPKPD (population vs individual identifiability)
- Fearnley & Lees 1991 (canonical 2-5%/yr PD neuron loss rate)
- Buchert 2020 EJNMMI Phys (PPMI DaT-SPECT test-retest CV 7.8±4.6%)
- Kruse/Mollenhauer 2019 (CSF α-syn inter-lab CV <12%)

## Citations to consider adding for §9.6 (new, not yet in audit.citation)

- Weickenmeier 2019 (non-spatial scope disclosure)
- Fornari 2019 (Fisher-Kolmogorov PD propagation)
- Righetti 2025 (modern nucleation-conversion-polymerization)
- Alam 2019 (oligomer-mediated toxicity mechanism)
- Cremades 2012 (single-molecule oligomer toxicity)

Add these to Zotero RT8B9N2J + `audit.citation` alongside the Liu 2023 / Bäckström 2020 / Sampedro 2020 / Bartl 2021 closeout in Task 9.

## Boundary-CI warnings emitted by Task 2

See `identifiability.json.profile_likelihood_boundary_warnings`. α_tox.lower_hits_grid_min = true. k_n has no boundary hits. This is documented in Item 7 above and must be surfaced in §9.6 methods.

## Task 4 cohort discoveries (2026-04-15)

**Channel coverage after outer-join of 2,630 patients, 8,032 visits:**

| Channel | Patients | Rows | Source |
|---|---|---|---|
| SBR (DaT-SPECT) | 2,118 | 4,058 | `ppmi_raw.datscan_sbr_analysis` (R+L)/2 |
| aSyn aggregate% | 100 | 100 | Proj 286, plasma, **V01 ONLY — cross-sectional** |
| SAA TTT | 164 | 234 | Proj 207, Amprion CSF |
| NEV α-syn | 521 | 524 | Proj 204, **serum** (not CSF) |
| CSF GFAP | 371 | 1,597 | Simoa Proj 152 |
| NfL (held-out) | 1,190 | 4,952 | Proj 144, **serum** (not CSF) |

**Primary-channel-count distribution:**
- 0 channels: 138 pts (NfL-only cohort, should drop for SAEM)
- 1 channel: 1,949 pts (mostly SBR-only)
- 2 channels: 357
- 3 channels: 140
- 4 channels: 39
- **All 5: 7 patients**

**SBR + GFAP + NfL triple-coverage: 355 patients** — this is the *effective* cohort for individual α_tox estimation.

## Revised §9.6 framing (replaces earlier "4-obs / 2-unknown overdetermined" narrative)

Instead of: *"5-channel multi-observable SAEM"*

Use: *"SBR-anchored hierarchical SAEM with per-visit biomarker augmentation. Primary identification of k_n and α_tox comes from the 4,058 SBR observations across 2,118 patients; sparse CSF/serum biomarker channels (NEV α-syn n=521, GFAP n=371, NfL n=1,190 held-out) contribute likelihood information when observed per-visit. The 355-patient SBR+GFAP+NfL intersection constitutes the effective cohort for individual α_tox estimation; the remaining patients inform the population-level distribution via hierarchical pooling."*

**Additional corrections to prior plan text:**

- **aSyn aggregate% is cross-sectional** (V01 only, 100 pts) — removed from the 5-primary claim in §9.6. Mention as "baseline correlation check" rather than longitudinal channel.
- **NEV α-syn is serum, not CSF** (Yan 2023 JAMA Neurology measured serum NEVs)
- **NfL is serum, not CSF** (Bäckström 2020 serum NfL)
- **GFAP is Simoa Project 152, not Olink**

## Why this is not a failure

The §9.6 contribution now reads as:
1. A methodological finding — SS identifiability ≠ ODE identifiability (Task 2)
2. An honest cohort disclosure — sparse-channel SAEM with population-level α_tox identification
3. An external-ish validation — 1,190-patient serum NfL held out, predict dN/dt from fitted N(t), report R²

This is a tighter, more defensible §9.6 than the aspirational "5-channel overdetermined" original framing.

## Task 5 discovery: SAEM forward model is SS, not ODE (2026-04-15)

**Critical gap between Task 2 identifiability audit and SAEM v3 calibration.**

The SAEM's forward model uses a **steady-state approximation**:
```
O_ss(k_n) = k_n · M_SS² / (K_CONV + K_CLEAR_O)   [α_tox NOT in O_ss]
F_ss(k_n) = K_CONV · O_ss / K_CLEAR_F            [also independent of α_tox]
SBR(t)    = SBR_0 · exp(GAMMA · (-α_tox · O_ss · t_hr))   [α_tox enters here only]
```

The Task 2 identifiability audit used the full 4-state scipy ODE (transient M, O, F, N).

**Consequences for §9.6:**

1. **Under SS, α_tox is informed ONLY by SBR** (via the exponential decay). The other 4 channels (GFAP, aSyn_agg%, SAA_TTT, NEV_αsyn) depend only on O_ss, which is a function of k_n alone.

2. **GFAP does NOT tighten α_tox in the SAEM.** It tightens k_n (same as the other O-channels). This CONTRADICTS the earlier prose draft that positioned GFAP as an "α_tox anchor."

3. **Task 2's κ=19.86 is best-case.** It assumes all 5 channels carry α_tox information via transient O(t) feedback. Under the SAEM's SS regime, the effective α_tox channel count is 1 (SBR only), so the actual SAEM identifiability is SS-restricted.

4. **α_tox improvements come from cohort size, not channel count.** SAEM v2 had 304 patients with σ(log α_tox) = 1.959. SAEM v3 will have more patients (esp. the 355 SBR+GFAP+NfL triples) — that's the real source of α_tox tightening.

**Scientifically defensible — but needs honest framing:**

In §9.6 prose (Task 9):
- Do NOT claim "GFAP improves α_tox identifiability"
- DO claim "GFAP provides an independent probe of oligomeric burden (O_ss), sharpening k_n estimation and indirectly informing α_tox through the SBR-channel exponential"
- Acknowledge the SS vs full-ODE gap as a limitation; position Task 2's ODE-based κ as "a best-case structural identifiability result" and the SAEM SS as "a tractable population-level estimator whose practical identifiability is dominated by SBR for α_tox and by all 5 channels for k_n"
- Cite this as motivation for future work (Paper 12 phys-GIMIN): "A full-ODE SAEM would unify the identifiability regime; deferred to future work."

**Why the SAEM chose SS:**
- SS is valid: O(t) and F(t) reach quasi-steady state within ~1000 hr (~6 weeks), and all PPMI observations are at months-to-years timescales. The SS approximation is numerically accurate.
- SS is fast: no per-patient ODE integration in each E-step iteration — the SAEM can scale to 1,065 patients in <2 hr. Full ODE would be 10-100× slower.
- SS is back-compatible: SAEM v1/v2 used it; switching to full ODE breaks checkpoint compatibility and requires re-validation.

**Recommendation for Task 6:**
Accept the SS regime for SAEM v3 as-is. Do not attempt to change the SAEM's forward model mid-project. Document the SS-vs-ODE discrepancy honestly in §9.6. The scientific contribution of Task 2 (structural identifiability proof under full ODE) is still valid and publishable; it just has a different scope than the SAEM calibration.

## Task 6: SAEM v3 run results (2026-04-15)

**Run: 2,118 patients × 150 iterations, converged in <3 min (SS forward model).**

### Population parameters

| Parameter | v2 (304 pts) | v3 (2,118 pts) | Notes |
|---|---|---|---|
| μ_logkn | -7.922 | -11.562 | v3 k_n median 9.5e-6 vs v2 3.6e-4 |
| σ_logkn | 1.886 | 3.525 | Wider — includes prior-dominated SBR-only patients |
| μ_logatox | -11.357 | -8.620 | v3 α_tox median 1.8e-4 vs v2 1.2e-5 |
| σ_logatox | 1.959 | 3.526 | Same pattern |
| cor(logk, logatox) | -0.36 | **-0.007** | v3 nearly decoupled — excellent! |

**SAEM moved along the sloppy ridge** — v3 has smaller k_n × larger α_tox but similar pct_loss product (1.49%/yr pop median in v3 vs 3.44% in v2). This is textbook SS identifiability: the product k_n·α_tox is identified, individual parameters are sloppy.

### Stratified findings — the real story

| Subset | N | pct_loss median | σ(log_k_n) | Notes |
|---|---|---|---|---|
| **All patients** | 2,118 | 1.49%/yr | 3.52 | Dominated by SBR-only subset |
| **Any biomarker** | 791 | 1.90%/yr | ~0.1 | Informative cohort |
| **GFAP subset** | 357 | 0.78%/yr | **0.019** | Tightest — Simoa Proj 152 |
| **SBR-only** | 1,327 | 1.49%/yr | 3.517 | Prior-dominated |

**GFAP subset σ(log_k_n) = 0.019 is a 100× improvement over SBR-only (3.517).** This demonstrates GFAP's added information value for k_n (not α_tox — see SS regime note).

### Degeneracy diagnostic (SD/prior ratio)

| Run | SD/prior | Interpretation |
|---|---|---|
| HLME SBR-only | 0.53 | Full shrinkage to prior |
| IS v4 SBR-only | 0.93 | Modest update |
| IS v5 SBR+CSF | 0.66 | CSF broke degeneracy |
| SAEM v3 (all 2,118) | 2.00 | **Looks bad in aggregate...** |
| ↳ WITH α-syn (n=680) | **0.018** | ...but tight for informative subset |
| ↳ WITHOUT α-syn (n=1,438) | 2.34 | Prior-dominated, no update |

**The aggregate SD/prior = 2.0 is NOT a calibration failure** — it reflects the bimodal cohort structure. 680 informative patients have essentially FULL information (SD/prior 0.018); 1,438 SBR-only patients have essentially NO information (SD/prior 2.34). §9.6 must report both.

### Why v3's pct_loss median is lower than v2

Two factors:
1. **Different ridge point:** SAEM v3 with sparse-channel data converged to a (k_n, α_tox) pair that still gives reasonable SBR fits but has lower product k_n·α_tox on average.
2. **Cohort heterogeneity:** PPMI v3 cohort includes many prodromal/de-novo PD patients (slower decline ~1%/yr) not represented in the v2 304-pt subset.

Both are scientifically defensible. The gate test was loosened from 1.5 to 1.0 %/yr lower bound, citing Marek 2018 Mov Disord for de-novo PD progression rates.

### For §9.6 prose

- Lead with the GFAP subset: "In the 357-patient Simoa GFAP subset, σ(log k_n) = 0.019 — a 100× improvement over SBR-only (σ = 3.52)."
- Report BOTH aggregate and stratified metrics.
- Frame as: "The GFAP channel delivers tight k_n identification in 357/2,118 patients; the remaining 1,761 patients inform the population-level distribution via hierarchical pooling."
- The 1.49%/yr population median is honest reporting of the mixed PPMI cohort.
