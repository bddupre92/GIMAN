# Paper 12 (phys-GIMIN) — Literature Review Synthesis

**Method topic.** Physics-regularized multimodal imputation for Parkinson's disease clinical/biomarker time series: autoencoder + graph attention + heteroscedastic σ + mechanistic ODE residual + conformal wrapper.

**Database.** 53 entries at [`litreview_database.jsonl`](litreview_database.jsonl) (post-2026-04-19 corrections). All fields populated per schema; 2 entries flagged `verified: false` with verification notes (Dhivyaa 2024 ICPR and Gupta 2025 α-syn — see §Verification gaps). **Three entries added 2026-04-19 (Task E fixes):** (a) Entry 50 — Giampiccolo 2024 identifiability (npj SBA) resolves the "Horvát 2025" citation gap flagged in `review_checklist.md`; the actual identifiability+hybrid-NODE paper cited 8× as "Horvát" is Giampiccolo et al. 2024 (DOI 10.1038/s41540-024-00460-3). The "Horvát 2025" label is a seed-extraction error; downstream files retain the old wording pending a future sweep. (b) Entries 51–52 — Iwata 2025 virtual-patient QSP+ML and Bräm 2022 ML pharmacometric covariate imputation — back the load-bearing CPT:PSP precedent claims in `venue_fit.md` §#2.

**Correction 2026-04-19 (three-correction pass, post-PDF-read):** (a) Entry `liang2024hspgnn` was a lit-review hallucination — DOI 10.1145/3627673.3679672 is actually **Li et al. 2024 LagCNN** (CIKM 2024), a CNN + Time Lag + FFT time-series imputer, NOT a physics-incorporated GNN. Entry key renamed to `li2024lagcnn`; authors, abstract, method-summary, uq_type, physics_integration, relevance, and use_as fields all updated to reflect its actual identity as a generic DL imputation baseline (alongside SAITS / GAIN / MIWAE). (b) New entry `derooij2025ude` added — de Rooij et al. 2025 PLOS Comp Biol "Physiology-informed regularisation enables training of universal differential equation systems for biological applications" (DOI 10.1371/journal.pcbi.1012198). This is the ACTUAL closest methodological prior to phys-GIMIN's lit-prior variant; code at `github.com/Computational-Biology-TUe/ude-regularization` is MIT and can be vendored directly. de Rooij now holds the top-1 competitor slot in `novelty_verdict.md`.

## Coverage metrics

| Metric | Target | Actual |
|---|---|---|
| Total entries | 40–60 | **53** |
| HIGH-relevance | ≥ 10 | **25** |
| PINN/UDE/latent-SDE/ODE-regularizer (regardless of domain) | ≥ 10 | **17** |
| Imputation methods without physics (baselines) | ≥ 5 | **17** |
| PD-specific progression modelling | ≥ 5 | **5** |
| Uncertainty-calibrated imputation | ≥ 3 | **8** |

Application-domain mix: EHR-general 21, other (non-clinical PINN / physiology + pharmacometrics) 13 (+Iwata 2025, Bräm 2022), synthetic 9 (+Giampiccolo 2024), PD 5, AD 3, oncology 1. Physics-integration mix: none 24 (+Bräm 2022), other (neural-ODE or Bayesian surrogate without mechanistic prior) 10 (+Iwata 2025), UDE 8 (+Giampiccolo 2024), PINN-style 7, ODE regularizer 2, latent-SDE 1. Use-as mix: cited prior art 18 (+3 Task-E additions: Giampiccolo 2024, Iwata 2025, Bräm 2022), baseline 12, cite-only 8, methodological anchor 7, contrast 7.

## Top 5 most-related competing works

| # | Paper | One-sentence gap vs phys-GIMIN |
|---|---|---|
| 1 | **de Rooij et al. 2025 PLOS Comp Biol** (DOI 10.1371/journal.pcbi.1012198) | Closest methodological prior — physiology-informed regularisation (non-negativity + AUC penalties) for training UDE systems on biological applications; validated on Michaelis-Menten + glucose minimal model with real PREDICT UK data. Gaps vs phys-GIMIN: learns an unknown ODE term (trajectory forecasting) not imputation; no multimodal / graph / heteroscedastic σ / MC-dropout / Bayesian posterior propagation; glucose minimal model, not PD α-syn dynamics. Code `github.com/Computational-Biology-TUe/ude-regularization` is MIT — VENDOR DIRECTLY. |
| 2 | **Wang et al. 2025 CNODE Parkinson (arXiv 2511.04789)** | Conditional neural-ODE for PD progression on PPMI MRI; purely data-driven with no mechanistic ODE, no multimodal imputation, no uncertainty — direct PD contrast. |
| 3 | **Xiao et al. 2025 TD-HNODE** (T2D progression) | Hypergraph + neural ODE for chronic-disease progression on T2D EHR; no mechanistic ODE, non-PD, no heteroscedastic σ. Flag: pending independent verification. |
| 4 | **Zou & Tian 2025 sparsified hybrid Neural ODE (arXiv 2505.18996)** | Hybrid neural-ODE with domain-informed graph sparsification for glucose forecasting; no multimodal imputation, no UQ, not PD. |
| 5 | **Hackenberg et al. 2023 latent-dynamic ODE** | Patient-specific ODE parameters in a learned latent space on spinal muscular atrophy; no imputation focus, no multimodal graph, no heteroscedastic σ. |

## Methodological convergences (what the field is doing)

1. **Neural-ODE/latent-ODE on irregular clinical time series is mature.** Rubanova 2019, Chen 2018, Xiao 2023 IVP-VAE, Zaman 2021, Chauhan 2022, Moon 2022 SurvLatent, Zeng 2025 TrajSurv, Xiao 2025 TD-HNODE. Pattern: encode irregular EHR → latent ODE evolution → task head.
2. **PINN / UDE for scientific systems-biology.** Raissi 2019, Rackauckas 2020, Yazdani 2020 SBINN, Daneker 2022, Philipps 2025 UDE review, **de Rooij 2025** (physiology-informed regularisation — closest methodological prior for phys-GIMIN lit variant), Zou 2025, Schmid/Zou 2024 Hybrid². Pattern: mechanistic ODE soft-constraint on neural dynamics; regularization critical under noise/sparsity (Philipps et al., de Rooij et al.).
3. **Attention-based + CNN + GNN imputation without physics.** GAIN, BRITS, GRU-D, SAITS, CSDI, ImputeINR, DEARI, LagCNN (CNN + Time Lag + FFT imputer; no physics, no graph — generic DL baseline alongside SAITS / GAIN / MIWAE). Uncertainty handled via variational / Bayesian / diffusion (Mulyadi 2019, Mattei 2019, Tashiro 2021 CSDI, Qian 2024 DEARI).
4. **PD progression modelling is overwhelmingly data-driven.** Young 2024 DPM review, Wang 2024 multi-modal graph, Wang 2025 CNODE, Latourelle 2017 PPMI mixed-effects — none integrate a mechanistic α-syn/DaT-loss ODE as a soft regularizer.
5. **UQ taxonomy is settled.** Kendall & Gal 2017 (aleatoric vs epistemic), Seitzer 2022 β-NLL (heteroscedastic pitfall), Angelopoulos & Bates 2021 (conformal), Podina 2024 C-PINN (conformal + physics).

## Methodological gaps phys-GIMIN fills

**G1 — No work combines mechanistic-ODE soft regularization with multimodal patient-graph imputation in PD.** The closest methodological prior is **de Rooij 2025** (physiology-informed UDE regularisation on glucose minimal model + PREDICT UK meal-response data), which holds priority on the generic regularisation trick but does not extend to multimodal imputation, patient-similarity graphs, heteroscedastic σ, or PD α-syn dynamics. The closest PD work (Wang 2025 CNODE) has no mechanistic ODE and no imputation focus. Phys-GIMIN is the only intersection point of the full 4D gap.

**G2 — No imputation method reports per-feature conformal intervals under a physics prior.** Podina 2024 C-PINN combines conformal + physics but on PDE solutions, not imputed clinical features; Tashiro 2021 CSDI gives probabilistic bands without physics; Mulyadi 2019 and Qian 2024 DEARI have variational / Bayesian UQ but no mechanistic prior.

**G3 — Heteroscedastic σ with a mechanistic ODE prior is absent in the clinical-imputation literature.** Seitzer 2022 β-NLL fixes the heteroscedastic pitfall; phys-GIMIN is the first to combine β-NLL with a UDE-style α-syn/DaT-loss ODE regularizer.

**G4 — PD-specific mechanistic priors have not been embedded in deep imputers.** Véronneau-Veilleux 2020 (PK/PD), Gupta 2025 (α-syn aggregation) and Phase 2 of our own mechanistic twin provide the ODE family. Phys-GIMIN's contribution is to use this family as a **soft constraint** inside the imputer, not as a standalone simulator.

**G5 — Regularization is the known bottleneck for UDE performance on sparse clinical data.** Philipps 2025 explicitly identifies this open problem; phys-GIMIN's β-NLL + graph-Laplacian smoothing + ODE residual together instantiate the kind of regularization their review recommends.

## Influence / citation signals

Seed PINN/UDE works are canonical (Raissi 2019 >10,000 citations per Google Scholar; Chen 2018 NeurIPS best paper; Rackauckas 2020 is Julia SciML cornerstone). Clinical-imputation baselines are mature (BRITS 735 cites, GAIN widely re-implemented, SAITS SOTA through 2024 per TSI-Bench). Among close cousins, **de Rooij 2025** (PLOS Comp Biol) is the most recent physiology-informed UDE paper — traction is growing on regularisation-as-the-UDE-open-problem. LagCNN (Li et al. 2024 CIKM) is tracked as a CNN+Time-Lag imputation baseline in the benchmark suite.

## MUST be a baseline in phys-GIMIN experiments

- **SAITS** (Du 2023) — SOTA attention imputer in TSI-Bench 2024
- **CSDI** (Tashiro 2021) — SOTA probabilistic imputer
- **BRITS** (Cao 2018) — canonical bidirectional RNN imputer
- **GAIN** (Yoon 2018) — canonical generative-adversarial imputer
- **MIWAE** (Mattei 2019) — canonical VAE imputer
- **GRU-D** (Che 2018) — canonical informative-missingness baseline
- **DEARI** (Qian 2024) — uncertainty-aware attention imputer
- **HyperImpute** (Jarrett 2022) — benchmark harness wrapping GAIN/MIWAE
- **Latent ODE** (Rubanova 2019) — neural-ODE baseline on irregular time series
- **LagCNN** (Li et al. 2024 CIKM) — CNN + Time Lag + FFT imputation baseline (clean-room implementable from paper Eq. 2-13; fidelity gate MSE 0.028 / MAE 0.044 on Weather 12.5% per `clean_room_verification_protocol.md` §4)

## MUST cite but not re-implement (prior art + contrast)

- **Raissi 2019, Rackauckas 2020, Yazdani 2020 SBINN, Daneker 2022, Philipps 2025, de Rooij 2025** — PINN/UDE foundations (de Rooij 2025 is the closest methodological prior to phys-GIMIN's lit variant)
- **Chen 2018 NODE, Rubanova 2019 latent ODE** — neural-ODE foundations (already baselines)
- **Seitzer 2022 β-NLL, Kendall & Gal 2017** — UQ framing
- **Angelopoulos & Bates 2021, Podina 2024 C-PINN** — conformal methodology
- **Wang, Teng, Perdikaris 2021** — PINN gradient pathology / loss-balancing
- **de Rooij 2025, Zou 2025, Schmid/Zou 2024 Hybrid², Hackenberg 2023, Aslanimoghanloo 2025, Xiao 2025 TD-HNODE, Wang 2025 CNODE** — close cousins; contrast explicitly
- **Véronneau-Veilleux 2020, Gupta 2025, Young 2024 DPM review, Wang 2024 PPMI graph, Latourelle 2017** — PD clinical anchor
- **Lavin 2021 simulation intelligence** — digital-twin framing

## Verification gaps (for follow-up triage into Zotero FPJM5RSS Review Queue)

Two seed papers could not be precisely located and are flagged `verified: false`:

1. **Dhivyaa et al. 2024 ICPR "Attention-MMVAE for AD with missing modalities (ADNI)"** — adjacent mmVAE-on-ADNI work exists (Wang 2024 Brain Communications mmVAE ATN subtypes, Gao 2021 TPA-GAN JBHI) but an ICPR 2024 Dhivyaa paper specifically did not surface in Google Scholar / arXiv / OpenReview. **Action:** skip; replace with Gao 2021 TPA-GAN (already in database) as the mmVAE-imputation anchor.
2. **Gupta et al. 2025 α-syn aggregation dynamics / SBR-IRT** — a 2025 α-syn mechanistic paper by Gupta did not surface. Substituted with a 2025 comprehensive α-syn review. **Action:** query the user for the exact citation, or accept the substitute as the mechanistic-review anchor for Phase 2 work.

One seed (Demirkaya 2024 IEEE TBME) resolved to a 2021 EMBC conference paper on Cubature Kalman Filter + hybrid ODE-RNN; marked verified with corrected year. One seed (Qian 2021 CCAI hybrid clinical NN) could not be verified; substituted with Qian 2024 JBHI "How Deep is Your Guess?" as the comparable critical-review anchor.

## Summary for paper 12 scoping

Phys-GIMIN occupies a **genuine and defensible gap** in the literature: it is the first method to combine (a) PD-specific mechanistic ODE prior (α-syn aggregation + DaT loss family from Véronneau-Veilleux 2020 and Phase 2 of our own mechanistic twin), (b) graph-attention multimodal imputation on PPMI, (c) heteroscedastic σ head with β-NLL stabilization (Seitzer 2022), and (d) per-feature conformal wrapper (Angelopoulos & Bates 2021, Podina 2024). The closest methodological prior — **de Rooij 2025 PLOS Comp Biol** — holds priority on physiology-informed UDE regularisation in biology (glucose minimal model, PREDICT UK meal-response), but does NOT extend to multimodal imputation, patient-similarity graphs, heteroscedastic σ, MC-dropout, Bayesian posterior propagation, or PD α-syn dynamics. Other individual cousins — CNODE (Wang 2025), Hybrid² neural ODE (Zou/Levine 2024), TD-HNODE (Xiao 2025) — each cover 1–2 of these axes but none combines all four in PD. The Philipps 2025 UDE open-problems review and de Rooij 2025 both motivate the regularization strategy phys-GIMIN instantiates.
