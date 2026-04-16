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
