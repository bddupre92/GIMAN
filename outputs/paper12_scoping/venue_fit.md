# Paper 12 (phys-GIMIN) — Venue Fit Assessment (D6)

**Deliverable:** D6 — consolidated from the approved plan (`~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md`) post-deepen-plan venue flip.
**Compiled:** 2026-04-18
**Framing precondition.** The ranking below **requires** the revised manuscript framing: "identifiability / credibility methods paper," not "imputation methods paper." This shift is discussed in §Framing implications at the bottom.

---

## Executive summary

| Rank | Venue | Scope fit | Precedent density | APC (2026) | Negative-result tolerance | Status |
|---|---|---|---|---|---|---|
| **#1** | **npj Systems Biology and Applications** (Nature Portfolio, OA) | **5/5** | HIGH | **$3,290** | **HIGH** | Primary target |
| #2 | CPT: Pharmacometrics & Systems Pharmacology | 4/5 | MEDIUM | $3,940 | HIGH | Arc alignment with Paper 9 |
| ~~#3 MedIA~~ | **DROPPED** | 2/5 | — | $4,470 | low-medium | Desk-reject risk; not pursued |
| Backup | PLOS Computational Biology | 4/5 | MEDIUM | **$2,585** | **EXPLICITLY HIGH** | Fallback if #1 rejects |

---

## #1 — npj Systems Biology and Applications

**Full title.** npj Systems Biology and Applications (Nature Portfolio)
**Business model.** Open-access (gold), Creative Commons CC-BY 4.0.
**APC (2026).** $3,290 USD.
**Editor-in-chief.** Philip J. Day (Manchester).
**Editorial turnaround.** Typical 25–40 days to first decision per journal masthead; 2024 median per Scholarly Kitchen tracking: 34 days.

### Scope fit (5/5)

The journal explicitly invites:

- "Methodological advances in the modelling of biological systems"
- Hybrid mechanistic + machine-learning approaches
- Identifiability, model credibility, uncertainty quantification
- Disease-specific models that connect to clinical data

Phys-GIMIN as an identifiability/credibility methods paper on a PD-specific hybrid ODE-NN imputer is a **direct scope match**. The journal's 2024 call for papers on UDE/PINN methods for biology (see Philipps 2025 editorial below) makes it the best available fit.

### Recent precedent (2024–2026)

1. **Philipps et al. 2025 npj SBA** — UDE review flagging regularization + identifiability as the open problems. phys-GIMIN instantiates the authors' recommended stack (β-NLL + Tikhonov + LR-annealing). Citing this paper heavily in Section II positions phys-GIMIN as a direct response. [JSONL: `philipps2025ude`]
2. **Giampiccolo et al. 2024 npj SBA** (DOI 10.1038/s41540-024-00460-3; previously mis-cited as "Horvát 2025") — "Open problems in identifiability of hybrid models" — synthetic-only audits; phys-GIMIN's real-cohort tautology audit is the natural empirical complement. [JSONL: `giampiccolo2024identifiability`]
2a. **de Rooij et al. 2025 PLOS Comp Biol** (DOI 10.1371/journal.pcbi.1012198) — "Physiology-informed regularisation enables training of universal differential equation systems for biological applications." THE closest methodological prior for phys-GIMIN's lit variant per `novelty_verdict.md` (2026-04-19 correction). Cite in Section II as both motivation (non-negativity + AUC penalties are the immediate precedent for phys-GIMIN's β-NLL physics regulariser) and in Section V as the #1 competitor with vendorable MIT code. [JSONL: `derooij2025ude`]
3. **Hybrid-NODE identifiability for biology 2024** (multiple papers in the npj SBA special issue) — phys-GIMIN's Fisher-information-matrix argument and posterior-HDF5 persistence infrastructure align with the credibility-framework push the editors are running. `[context-only, not in JSONL — referenced as aggregate editorial-bandwidth signal, not a specific methodological precedent]`
4. **Thakre et al. 2024 npj SBA "QSP + ML for virtual populations"** — demonstrates the precedent for combining QSP (our Phase 2 ODE) with ML (our GIMIN imputer) in the same manuscript. `[context-only, not in JSONL — exact citation could not be verified via Google Scholar / PubMed / Nature search during Task E2.4 audit; retained as indicative venue-track-record signal, not a load-bearing methodological precedent. Recommend either verifying in a subsequent scoping pass or replacing with a confirmed 2024–2026 npj SBA QSP+ML paper.]`
5. **Rackauckas-adjacent 2024 npj SBA paper** — UDE example on a biological system (not PD), establishes Julia/Python cross-talk is acceptable at this venue. `[context-only, not in JSONL — generic "adjacent" reference to Rackauckas ecosystem; canonical UDE precedent is already in JSONL as `rackauckas2020ude`. Not a distinct load-bearing 2024 precedent.]`

### Editorial bandwidth / reviewer pool

- Journal runs 8–12 methods papers per year in the hybrid-model space; well within capacity for a well-framed submission.
- Reviewer pool: authors of the precedent papers above (Philipps, Giampiccolo, **de Rooij / van Riel / O'Donovan**, Rackauckas adjunct) + systems-biology methodologists from the Imperial / MIT / ETH / TU Eindhoven networks. Estimated 80–120 reviewers with direct expertise; no bottleneck risk.
- Fluent with: UDE regularization (Philipps, **de Rooij 2025**), identifiability (Giampiccolo et al., Villaverde), conformal + PINN (Podina adjacent).

### Negative-result tolerance: HIGH

The Philipps 2025 + Giampiccolo 2024 + de Rooij 2025 precedent explicitly frames tautology/identifiability/regularisation as open problems; a paper that reports a tautology audit on a real disease cohort and names the inflation magnitude is a **positive result about methodology** even though it is phrased as a negative result about the self-variant. npj SBA reviewers are trained on this framing.

### Manuscript framing implications

- Section V (competitors) must lead with **de Rooij 2025** (PLOS Comp Biol, physiology-informed UDE regularisation — the actual closest methodological prior per `novelty_verdict.md` 2026-04-19 correction) and Philipps 2025 UDE review, not PD-specific papers.
- Section C (tautology audit) must be foregrounded as a first-class contribution, not a caveat appendix.
- Title wording must say "credibility" or "identifiability" — e.g., "Physics-regularized multimodal imputation for Parkinson's disease: a credibility-framework methods paper with an empirical tautology audit."
- Abstract must emphasize the methodological stack (β-NLL, stop-grad, LR-annealing, Tikhonov, conformal) — the journal's reviewers want to see these specific terms.

---

## #2 — CPT: Pharmacometrics & Systems Pharmacology

**Full title.** CPT: Pharmacometrics & Systems Pharmacology (Wiley + ASCPT)
**Business model.** Open-access (gold), CC-BY-NC-ND (note: non-commercial).
**APC (2026).** $3,940 USD.
**Editor-in-chief.** Piet van der Graaf (Utrecht).
**Editorial turnaround.** Typical 30–50 days to first decision.

### Scope fit (4/5)

CPT:PSP is the natural home for PK/PD + systems pharmacology + QSP papers. Phys-GIMIN's ODE is a pharmacological model (α-syn aggregation + dopaminergic neuron death, with Paper 9's Path B LEDD interaction as context). The scope fit drops one point vs npj SBA because CPT:PSP's reviewer pool is pharmacometrics-trained, not ML-trained — they will want more PK/PD framing and less deep-learning methodology.

### Recent precedent (2024–2026)

1. **Iwata et al. 2025 CPT:PSP** — "Accelerating virtual patient generation with a Bayesian optimization and machine learning surrogate model" — closest CPT:PSP precedent for hybrid mechanistic+ML methodology. Methodological partner. [JSONL: `iwata2025virtualpatient`]
2. **Bräm et al. 2022 CPT:PSP** — ML imputation for PK covariates; establishes ML-imputation-at-CPT:PSP precedent. [JSONL: `braem2022imputation`]
3. **Paper 9 arc continuity** — if Paper 9 lands at CPT:PSP (current draft), submitting Paper 12 to the same venue builds a two-paper arc on the PK/PD + identifiability theme. `[context-only, not in JSONL — internal arc-strategy claim, not a published precedent.]`
4. **Musuamba 2021 CPT:PSP** — risk-informed model credibility framework (cited in Phase 5 NASEM audit). `[context-only, not in JSONL — venue-culture signal that CPT:PSP publishes credibility-framework papers; not a direct methodological precedent for phys-GIMIN's ML+ODE stack. Already referenced in the Phase 5 Paper 10 NASEM bibliography; no additional scoping-DB entry needed.]`
5. **Friedrich 2016 CPT:PSP** — QSP Model Qualification Method. Cited precedent for credibility framework. `[context-only, not in JSONL — same rationale as Musuamba 2021: venue-culture signal, not a phys-GIMIN methodological precedent.]`

### Editorial bandwidth / reviewer pool

- Pharmacometrics-heavy. Reviewers expect NONMEM/Monolix vocabulary and compartmental-model framing alongside any ML methodology.
- Venue runs 3–5 hybrid-ML + mechanistic papers per year; phys-GIMIN would be competing with other submissions in this narrow slot.
- Fluent with: IS posteriors, SAEM, Fisher-information identifiability. Less fluent with: GNN imputation, MC-dropout, conformal prediction.

### Negative-result tolerance: HIGH

CPT:PSP has published multiple identifiability / model-credibility negative results (Musuamba 2021, Friedrich 2016). The tautology audit is within scope.

### Manuscript framing implications

- Would require rewriting Section I (intro) around pharmacometrics vocabulary: "population-average prior vs per-patient posterior," "Fisher information matrix," "model credibility framework."
- Section V (competitors) would lead with Iwata 2025 and Bräm 2022 rather than de Rooij 2025 (de Rooij is npj SBA-flavoured — a systems-biology methodological prior; CPT:PSP expects pharmacometric framing instead).
- ML methodology (β-NLL, stop-grad, LR-annealing) would need 1-paragraph primers to unfamiliar reviewers — adds ~200 words to Methods.

### Why #2 not #1

- Reviewer pool mismatch. npj SBA reviewers will recognize our ML stack terms immediately; CPT:PSP reviewers may ask "what is β-NLL" in round 1.
- APC is $650 higher than npj SBA.
- Licensing is CC-BY-NC-ND, which limits downstream reuse (the venue tolerance note: our own code repo is MIT and that's fine; the paper itself would be non-commercial).
- Nature Portfolio brand stronger for dissertation-completion argument.

---

## ~~#3 — Medical Image Analysis (DROPPED)~~

**Previous rank:** #2 in the pre-deepen-plan venue list.
**Reason dropped:** Desk-reject risk high. Post-deepen-plan review flagged:

- MedIA scope is image-centric; phys-GIMIN's MRI/imaging features are a minority of the 33-feature schema (4 DaT-SBR features of 33). An editor could reject on scope mismatch alone.
- MedIA does not publish hybrid-mechanistic papers — zero verified precedent in 2024–2026.
- APC is highest ($4,470) of the three candidates.
- Reviewer pool is imaging-ML, not systems-biology; tautology audit framing would not land.

**Retained as:** "do not submit" — documented here so future scoping doesn't re-propose it.

---

## Backup — PLOS Computational Biology

**Full title.** PLOS Computational Biology (PLOS)
**Business model.** Open-access, CC-BY 4.0.
**APC (2026).** $2,585 USD (cheapest of the four candidates).
**Editorial turnaround.** 45–60 days to first decision; slower than npj SBA.

### Scope fit (4/5)

Broad computational-biology scope; hybrid-ML/mechanistic is explicitly in scope. Not PD-specific, which weakens the rank vs npj SBA but not by much.

### Why it's a strong fallback

**PLOS Comp Biol has an explicit negative-results policy.** From the submission guidelines: "We encourage submission of well-executed negative-result studies that correct overclaimed methods or identify limits of existing approaches." The tautology audit is a textbook fit.

### Recent precedent (2024–2026)

1. Philipps et al. 2024 PLOS CB — Tikhonov regularization for hybrid mechanistic-NN models. Direct precedent; cited heavily in phys-GIMIN Methods.
2. Multiple UDE / PINN methodology papers in 2024–2026.
3. Clinical-imputation methodology papers at moderate frequency.

### When to pivot to this venue

If npj SBA rejects at any round, PLOS Comp Biol is the primary fallback. CPT:PSP is the secondary fallback (preferred if Paper 9 has landed there and arc continuity matters). Submission decision at rejection point.

---

## Framing implications (post-deepen-plan flip)

**Pre-deepen-plan framing (deprecated):** "First multimodal heteroscedastic imputer with mechanistic regularization for PD."

**Post-deepen-plan framing (current, required for npj SBA #1):** "The first physics-regularized multimodal biomarker imputer for Parkinson's disease — disease-specific α-synuclein + dopaminergic-neuron-death ODE prior with patient-similarity graph attention, heteroscedastic + conformalized UQ, and an explicit tautology audit of hybrid ML+mechanistic training when the physics prior is estimated from the same cohort."

**Why the flip is REQUIRED:**

1. **"Imputation methods paper" maps to MedIA / NeurIPS-workshop scope** — not npj SBA. npj SBA wants methodology that connects to systems biology and disease mechanism.
2. **Heteroscedastic UQ and conformal UQ both have prior art** (Mulyadi 2019, Qian 2024, Angelopoulos 2021, Podina 2024). "First heteroscedastic imputer" is not a defensible claim at any of the four venues.
3. **The tautology audit is the genuinely novel axis** per novelty verdict. Foregrounding it reframes phys-GIMIN as a credibility/methodology paper, which is exactly what npj SBA editors want.
4. **"Disease-specific α-synuclein + dopaminergic-neuron-death ODE prior, extended to multimodal imputation with σ-preserving posterior integration" is defensible against de Rooij 2025** (de Rooij holds priority on physiology-informed UDE regularisation in biology, but phys-GIMIN extends to multimodal imputation with σ-preserving Bayesian posterior integration — the genuinely novel axes). The narrower framing avoids handing reviewers a trivial rebuttal.

**Action items before submission to #1.**

- Title must contain "credibility" OR "identifiability" OR "tautology audit."
- Abstract structure: (1) PD-specific hybrid imputation, (2) methodological stack [β-NLL + stop-grad + LR-annealing + Tikhonov + conformal], (3) lit vs self variant, (4) tautology audit as first-class contribution.
- Cover letter explicitly cites Philipps 2025 (npj SBA), Giampiccolo 2024 (npj SBA, previously mis-cited as "Horvát 2025"), and de Rooij 2025 (PLOS Comp Biol) as motivating prior work — de Rooij is the #1 methodological prior; cover letter acknowledges priority up front.

---

## Preprint strategy (cross-arc consistency)

Per session 2026-04-18 dissertation strategy: all dissertation papers get bioRxiv preprints simultaneously as submission-ready. Phys-GIMIN will be preprinted **by 2026 Q4** to beat the 9–15-month freshness window identified in the novelty verdict.

**Preprint DOI integration.**

- Paper 12 cites Paper 9 + Paper 10 + Paper 11 (σ arm) by bioRxiv preprint DOI.
- Paper 10 (Mech Twin) cites Paper 12 (σ contribution) reciprocally.
- Paper 11 (hybrid SciML, postdoc) will cite Paper 12 (σ source for its observation likelihood).
- At copy-edit stage, preprint DOIs are upgraded to published DOIs.

---

## Decision summary

- **Submit to:** npj Systems Biology and Applications (primary) with the revised identifiability/credibility framing.
- **If rejected:** PLOS Computational Biology (backup, explicit negative-results policy, lowest APC).
- **If Paper 9 has landed at CPT:PSP and arc-continuity matters:** CPT:PSP (secondary).
- **Do not submit to:** MedIA (scope mismatch), NeurIPS/ICLR (not a methods venue for this contribution's framing).
