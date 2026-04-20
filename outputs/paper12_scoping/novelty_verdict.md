# Paper 12 (phys-GIMIN) — Novelty Assessment Verdict

```
VERDICT: CONDITIONAL
CONFIDENCE: MEDIUM
```

**Deliverable:** D3 scoping
**Compiled:** 2026-04-18
**Corrected:** 2026-04-19 — LagCNN/Liang mis-attribution resolved; de Rooij 2025 elevated to #1 methodological prior.
**Inputs used:** `litreview_database.jsonl` (53 entries after 2026-04-19 corrections; previously 52), `litreview_synthesis.md`, `github_inventory.md` (18 repos), approved plan `research-goal-onsider-using-jolly-matsumoto.md`.
**Method under assessment:** stage-aware multimodal graph autoencoder with heteroscedastic Gaussian decoder + MC-dropout, stage-conditioned attention bias + decoder embedding, ODE-regularized loss against the Phase-2 α-syn + neuron-death ODE, dual lit-prior / self-prior variants, per-feature temperature scaling + conformal wrapper, PPMI + BioFIND + PDBP cross-cohort evaluation, explicit tautology-audit negative-result section.

---

## Justification (≈520 words)

phys-GIMIN occupies a defensible but narrow 4D gap — (PD-specific mechanistic prior) × (multimodal patient graph imputation) × (heteroscedastic + conformal UQ) × (tautology audit) — that no single paper in the database simultaneously covers. The closest cousins each miss at least two of the four axes.

**(1) de Rooij et al. 2025 PLOS Comp Biol** is the single highest-risk competitor and the **actual closest methodological prior to phys-GIMIN's lit-prior variant**. It introduces physiology-informed regularisation (non-negativity + area-under-curve penalties) for training Universal Differential Equation (UDE) systems on biological applications, demonstrated on Michaelis-Menten synthetic data and on the glucose minimal model with real PREDICT UK meal-response measurements. This is exactly the same methodological pattern as phys-GIMIN's lit variant — a physiology-informed penalty term regularising a hybrid ML+mechanistic-ODE system. **However:** de Rooij learns an unknown ODE term via NN (trajectory forecasting), whereas phys-GIMIN imputes missing features with σ preservation; de Rooij has no multimodal / graph attention / heteroscedastic σ / MC-dropout / Bayesian posterior propagation; and de Rooij fits the glucose minimal model, not Parkinson's α-synuclein dynamics. phys-GIMIN extends the pattern to multimodal imputation with σ-preserving posterior integration — axes that remain genuinely novel after acknowledging de Rooij. Code at `github.com/Computational-Biology-TUe/ude-regularization` (MIT) is **vendorable directly**.

**(2) Wang 2025 CNODE PPMI (arXiv 2511.04789)** is the most dangerous PD-specific competitor. It is **conditional neural ODE for PD progression on PPMI MRI**. However, per the database entry, CNODE is **purely data-driven (no mechanistic prior)**, **MRI-only (not multimodal imputation)**, **no uncertainty**, and focused on **forecasting complete-data trajectories**, not imputation. phys-GIMIN's mechanistic ODE regularizer, multimodal biomarker scope, and imputation framing are not pre-empted.

**(3) Xiao 2025 TD-HNODE** (hypergraph + neural-ODE, T2D progression) demonstrates a cousin architecture on non-PD chronic progression — no mechanistic ODE, not multimodal, imputation is not the primary task. Flag as "pending independent verification" in the Loose-ends section: this entry was never directly fetched and confirmed during the lit-review pass.

**(4) Hackenberg 2023 latent-dynamic ODE** (SMA cohort) and **Aslanimoghanloo 2025 latent-SDE** (ICU cohort) each demonstrate cousin architectures on chronic-disease progression — neither uses a mechanistic ODE prior. **Demirkaya 2021 EMBC** (Cubature-Kalman hybrid ODE-RNN) and **Zou 2025** (sparsified hybrid neural ODE) operate on single-ODE physiological systems (retinal circulation, glucose) without multimodal imputation or UQ.

**Strongest individual-axis precedents:**
- UQ stack: β-NLL (Seitzer 2022) + Kendall-Gal 2017 + MC-dropout (Gal 2016) + conformal (Angelopoulos 2021, Podina 2024 C-PINN). phys-GIMIN composes these but is not the first on any one of them.
- UDE regularization: de Rooij 2025 explicitly instantiates the regularisation pattern phys-GIMIN is extending to imputation; Philipps 2025 flags the open problem more broadly.
- PD graph imputation: Wang 2024 PPMI graph paper is a graph progression model without mechanistic ODE or imputation focus.

**Why CONDITIONAL, not YES:** The composition argument is valid, but a single 2026 follow-up to de Rooij that extended the UDE regularisation pattern to multimodal clinical imputation would collapse the gap to cosmetic. The freshness risk is real. The tautology-audit negative-result section remains the strongest genuinely-novel axis.

**Note on LagCNN (Li et al. 2024 CIKM).** The lit-review agent originally labelled DOI `10.1145/3627673.3679672` as "Liang 2024 HSPGNN." That label was a hallucination — the actual paper at that DOI is **LagCNN** (Li, Jian, Wan, Geng, Fang, Chen, Gao, Jiang, Zhu), a CNN-based time-series imputer using Time Lag features + FFT. LagCNN is **not physics-regularised, not a graph neural network, and not clinical**. It is a generic imputation baseline alongside SAITS / GAIN / MIWAE — NOT a physics-regularised competitor to phys-GIMIN. Downstream files have been corrected. See `clean_room_verification_protocol.md` §4 for LagCNN's role as an ordinary DL imputation baseline.

---

## Top 3 strongest competing works (must appear in Paper 12 §V)

1. **de Rooij et al. 2025 PLOS Comp Biol** (DOI 10.1371/journal.pcbi.1012198) — physiology-informed regularisation for UDE (non-negativity + area-under-curve penalties) on Michaelis-Menten + glucose minimal model. Paper 12 §V must contrast: (a) trajectory forecasting vs multimodal imputation, (b) univariate ODE state vs 33-feature multimodal schema on a patient-similarity graph, (c) no UQ vs heteroscedastic σ + MC-dropout + per-feature conformal, (d) glucose minimal model vs PD α-syn + dopaminergic-neuron-death ODE. Open-source code at `github.com/Computational-Biology-TUe/ude-regularization` (MIT) can be **vendored directly** — no clean-room re-implementation needed for this competitor.

2. **Wang et al. 2025 CNODE PPMI (arXiv 2511.04789)** — the direct PD competitor on the same cohort. §V must make explicit that CNODE is (a) data-driven with no mechanistic prior, (b) MRI-only, (c) forecasting of complete data, NOT imputation, (d) no UQ. Frame as "complementary, not competing."

3. **Xiao et al. 2025 TD-HNODE** — hypergraph + neural-ODE on chronic-disease progression (T2D). §V must contrast the absence of mechanistic ODE and the T2D-vs-PD domain, and note that TD-HNODE's hypergraph structure could, in principle, be swapped into phys-GIMIN's graph module in future work. **Flag: this entry pending independent verification — the paper was never directly fetched during the 2026-04-18 lit-review sweep.**

(Secondary but required: Hackenberg 2023 latent-ODE on SMA, Aslanimoghanloo 2025 latent-SDE, Zou 2025 sparsified hybrid NODE, Demirkaya 2021 EMBC Cubature-Kalman hybrid ODE-RNN, Podina 2024 C-PINN.)

**Not in the top-3 competitor list** (moved to baseline status on 2026-04-19):

- **Li et al. 2024 LagCNN (CIKM)** — CNN + Time Lag + FFT time-series imputer. Originally mis-labelled by the lit-review agent as "Liang 2024 HSPGNN." LagCNN is a generic DL imputation baseline, not a physics-regularised competitor; it belongs alongside SAITS / GAIN / MIWAE in the benchmark suite (see `clean_room_verification_protocol.md` §4 for the LagCNN fidelity gate at MSE 0.028 / MAE 0.044 on Weather 12.5%).

---

## Framing recommendation

Re-evaluating the earlier "first multimodal heteroscedastic imputer with mechanistic regularization for PD" framing in light of **de Rooij 2025**, Wang 2025 CNODE PPMI, and the LagCNN mis-attribution correction, I recommend:

> **"First physics-regularized multimodal heteroscedastic imputer for PD with σ-preserving integration into Bayesian posterior updating — disease-specific α-synuclein + neuron-death ODE prior on patient-similarity graph, MC-dropout + conformal UQ, and an explicit tautology audit. de Rooij 2025 holds priority on physiology-informed UDE regularization in biology; phys-GIMIN extends the pattern to multimodal imputation with σ-preserving posterior integration — axes that remain genuinely novel after acknowledging de Rooij."**

This framing:

- **Narrows from "mechanistic regularization" to "σ-preserving multimodal imputation extension of de Rooij's regularisation pattern"** — defensible because de Rooij holds priority on the generic physiology-informed UDE regularisation trick, and phys-GIMIN's contribution is to extend it to the missing-feature imputation regime with heteroscedastic σ and per-feature conformal bands.
- **Foregrounds the negative-result section** as a first-class contribution, since Philipps 2025 UDE review and Giampiccolo 2024 (previously labelled "Horvát 2025") identifiability paper both flag identifiability/tautology but neither reports a disease-specific empirical negative result. This is the strongest genuinely-novel axis.
- **Drops the weaker "first heteroscedastic imputer" claim** — heteroscedastic UQ in clinical imputation exists (Mulyadi 2019, Qian 2024 DEARI, Kendall 2017). phys-GIMIN composes but does not invent.
- **Drops "first conformal imputer"** — Podina 2024 C-PINN already conformalized a physics-informed model; phys-GIMIN extends to imputation outputs but is not the first to compose conformal + physics.

The composition + PD specificity + σ-preserving-for-Bayesian-update + tautology audit is the defensible quadrant.

---

## Freshness risk

**High.** Three of the seven seed competitors (Xiao 2025, Wang 2025, Zou 2025) published in 2025; de Rooij 2025 (the new #1 methodological prior) also published in 2025. The field is compressing toward the same gap phys-GIMIN claims. Specific risks on a 6-12 month horizon:

1. **A 2026 de Rooij follow-up extending to multimodal clinical imputation.** de Rooij's PREDICT UK glucose model is a natural jumping-off point for a meal-response multimodal biomarker imputer. A clinical-domain version (PD, diabetes, sepsis) is the natural next paper. **Mitigation:** file phys-GIMIN preprint by Q3 2026; cite de Rooij extensively in §II and §V; vendor the `ude-regularization` code so we can run head-to-head.

2. **A 2026 CNODE + imputation extension.** Wang 2025 team has PPMI access and PyTorch stack; adding an imputation head is 2-3 months of work. **Mitigation:** phys-GIMIN's mechanistic ODE prior is the hard-to-replicate asset — CNODE would need to re-derive or adopt the Phase-2 α-syn+N(t) ODE.

3. **A 2026 UDE-for-neurodegeneration tutorial from the Rackauckas / SciML group.** Their Julia-first stack is one paper away from a neuro-specific UDE example. **Mitigation:** phys-GIMIN is PyTorch-native and imputation-focused; a Julia UDE demonstration on complete data would not displace it.

**Estimated window:** 9-15 months before a credible head-to-head PD competitor plausibly appears. Preprint by 2026 Q4 is prudent.

---

## Pivot triggers

The verdict flips from **CONDITIONAL to NO** if any of the following surfaces in the D4 literature expansion (open search, 2024-2026, targeting conferences and arXiv):

1. **A 2026 de Rooij follow-up that combines physiology-informed UDE regularisation with multimodal clinical imputation on any disease cohort** (glucose, PD, AD, sepsis). This is the exact intersection phys-GIMIN claims. Even one existing paper collapses the composition argument. Specific search queries: `("physiology-informed" OR "UDE regularization" OR "de Rooij") AND (imputation OR "missing data") AND (clinical OR biomarker)`; `("stage-conditioned" OR "NSD-ISS") AND (neural ODE OR hybrid ODE)`.

2. **A published tautology/identifiability audit for a hybrid ML+mechanistic model on a real disease cohort.** Philipps 2025 and Giampiccolo 2024 flag the problem but report only synthetic benchmarks. A real-cohort empirical tautology paper would collapse the negative-result novelty axis. Search: `("hybrid" OR "UDE" OR "PINN") AND (identifiability OR tautology OR "self-prior") AND ("clinical" OR "cohort" OR "real-world")`.

3. **A 2025-2026 paper that combines a disease-specific mechanistic ODE prior with multimodal graph imputation on PD or any neurodegenerative cohort.** Search: `("physics-informed" OR "mechanistic" OR "UDE") AND (imputation OR "missing data") AND (Parkinson OR PPMI OR "alpha-synuclein" OR "dopaminergic")`.

If zero hits on all three triggers after a 2-hour targeted search, upgrade the verdict to **YES (HIGH confidence)**.

---

## Verification flags (do not fabricate)

Two claims inherited from the synthesis are **unverified at entry level** and should be flagged in Paper 12 §V:

- **Dhivyaa 2024 ICPR mmVAE-ADNI** — seed citation; the paper did not surface in targeted search (`verified: false` in JSONL). Replace with Gao 2021 TPA-GAN as the mmVAE-imputation anchor.
- **Gupta 2025 α-syn aggregation** — seed citation; substituted with a 2025 MDPI α-syn review (`verified: false`). User should confirm the exact Gupta paper or accept the substitute.

One seed was **corrected**: Demirkaya 2024 IEEE TBME → Demirkaya 2021 IEEE EMBC (conference version verified via PubMed 34891402). Paper 12 §V must use the 2021 citation, not 2024.

**Critical correction (2026-04-19):** DOI `10.1145/3627673.3679672` was originally labelled "Liang 2024 HSPGNN" by the lit-review agent. That attribution was a hallucination. The ACTUAL paper at that DOI is **LagCNN** (Li, Jian, Wan, Geng, Fang, Chen, Gao, Jiang, Zhu 2024 CIKM), a CNN + Time Lag + FFT time-series imputer — NOT a physics-regularised GNN. LagCNN is reframed as a generic DL imputation baseline (alongside SAITS / GAIN / MIWAE); it is NOT in the top-3 physics-regularised competitor list. de Rooij 2025 takes the top-1 slot as the actual closest methodological prior.

Licensing status for clean-room baselines (`github_inventory.md` §1):

- `bobjz/H2NCM` (Zou 2025 precursor) — NO LICENSE. Clean-room re-implementable from paper text (arXiv 2505.18996v3 full algorithm available).
- `neu-spiral/Hybrid-ODE-NN` (Demirkaya 2021 companion) — NO LICENSE. Clean-room re-implementable from paper text (Eq. 4-11 fully specified).
- `Computational-Biology-TUe/ude-regularization` (de Rooij 2025) — **MIT** (OSI-compatible). **VENDOR DIRECTLY** — no clean-room needed.

These do not affect the novelty verdict — novelty is about claims, not reproducibility infrastructure — but they do shape the Week-1 verification workload. Document in §VI implementation notes.

---

*End of verdict. All competitor papers cross-checked against `litreview_database.jsonl`; any claim in the narrative synthesis that could not be traced to a specific JSONL entry is flagged above.*
