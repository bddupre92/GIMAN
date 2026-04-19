# Paper 12 (phys-GIMIN) — Clean-Room Baseline Verification Protocol

**Deliverable:** D (Revision 3, per scoping plan §Revisions before postdoc execution)
**Compiled:** 2026-04-19
**Updated:** 2026-04-19 — §4 filled with extracted metrics after PDF reads; §5 email templates removed (no longer needed); §7 license risk table updated.
**Status:** Gate document. Any baseline listed here that fails §3 Step 4 is DROPPED from Paper 12 §V and recorded as a "cannot-reproduce" Related-Work note.

## §1 — Purpose

Four Paper 12 §V baselines require clean-room re-implementation from paper text because either (a) their companion repos have no LICENSE file (all-rights-reserved under GitHub ToS §D.5) or (b) no public code release exists:

- **LagCNN (Li et al. 2024 CIKM)** — no public repo located. Clean-room from paper Eq. 2-13.
- **Wang 2025 CNODE PPMI** — no public repo located. Clean-room from arXiv 2511.04789 §II.B-II.D.
- **Demirkaya 2021 EMBC** — companion `neu-spiral/Hybrid-ODE-NN` has no LICENSE. Clean-room from paper Eq. 4-11.
- **Zou 2025** — companion `bobjz/H2NCM` has no LICENSE. Clean-room from arXiv 2505.18996v3 full algorithm.

Reviewers routinely reject unfair comparisons against re-implemented baselines ("your version is weaker than the original — of course you win"). The only defense is a documented fidelity gate: the re-implementation must reproduce the original paper's headline metric on the original dataset **within 10%** before being admitted to Paper 12 §V. Otherwise the re-implementation becomes a Related-Work citation, not a baseline.

**One competitor is vendorable directly (no clean-room needed):**

- **de Rooij et al. 2025 PLOS Comp Biol** — companion repo `github.com/Computational-Biology-TUe/ude-regularization` is **CC-BY licensed**. **VENDOR DIRECTLY** into `paper12_phys_gimin/baselines/derooij2025/`. No fidelity gate, no clean-room. de Rooij 2025 is the #1 methodological prior for phys-GIMIN's lit variant per `novelty_verdict.md` (2026-04-19 correction).

**Important scope clarification.** LagCNN (Li et al. 2024 CIKM) is a **generic DL imputation baseline**, NOT a physics-regularised competitor. It was originally mis-labelled as "Liang 2024 HSPGNN" by a lit-review agent hallucination. LagCNN belongs in the DL-imputer zoo alongside SAITS / GAIN / MIWAE, not in the physics-regularised competitor list. Its fidelity gate (below) is still required if we admit it as a baseline.

## §2 — Competitors requiring clean-room

| # | Citation | Paper retrieval path | Competitor category |
|---|---|---|---|
| 1 | **Li et al. 2024 LagCNN** — CIKM 2024 (DOI 10.1145/3627673.3679672) | CIKM 2024 proceedings (ACM DL via UND EZProxy). | Generic DL imputation baseline (NOT physics-regularised) |
| 2 | **Wang et al. 2025 CNODE PPMI** — arXiv 2511.04789 | arXiv PDF. | Direct PD competitor |
| 3 | **Demirkaya et al. 2021 EMBC** — PubMed 34891402 | IEEE Xplore via EZProxy. Companion `neu-spiral/Hybrid-ODE-NN` un-licensed; do NOT reference. | Hybrid ODE-NN baseline |
| 4 | **Zou et al. 2025** — ICLR 2026, OpenReview id `QBzFrjEF59` | OpenReview PDF + supplementary. Companion `bobjz/H2NCM` un-licensed; do NOT reference. | Hybrid neural ODE baseline |

**Not requiring clean-room (vendor directly):**

| # | Citation | Retrieval path | Action |
|---|---|---|---|
| 5 | **de Rooij et al. 2025** — PLOS Comp Biol (DOI 10.1371/journal.pcbi.1012198) | Open access PDF + `github.com/Computational-Biology-TUe/ude-regularization` (CC-BY) | **VENDOR DIRECTLY** — no clean-room, no fidelity gate, no email |

## §3 — Verification protocol per competitor

Four sequential steps; apply identically to each clean-room competitor. Log each step in `paper12_phys_gimin/baselines/<competitor>/verification_log.md`.

- **Step 1 — Read.** Download the paper + any supplementary material. Extract pseudocode or architecture spec into a `spec.md` inside the competitor sub-dir.
- **Step 2 — Re-implement.** Write the clean-room PyTorch implementation under `paper12_phys_gimin/baselines/<competitor>/`. Comments may cite the paper's equation numbers; comments may NOT reference any existing GitHub repo. Add a `LICENSE` (MIT) + a `README.md` that explicitly states "independently re-implemented from <paper citation>; no source derived from <upstream repo (if any)>." LOC / hour estimates:

  | Competitor | Est LOC | Est hours |
  |---|---|---|
  | LagCNN (Li et al. 2024) | 300–500 | 20–30 |
  | CNODE PPMI (Wang 2025) | 400–600 | 25–40 |
  | Demirkaya 2021 | 500–700 | 30–45 |
  | Zou 2025 | 400–600 | 25–40 |

- **Step 3 — Reproduce on original dataset.** Obtain the original dataset (or the closest public equivalent identified in the paper). Run the re-implementation under the paper's reported hyperparameters. Record the single headline metric.
- **Step 4 — Fidelity gate.** Compute `|ours − published| / |published|`.
  - **≤ 10%** → PASS → admit to Paper 12 §V baseline suite after integration with phys-GIMIN's 33-feature schema.
  - **> 10%** → FAIL → file a "cannot reproduce" entry in `paper12_phys_gimin/baselines/<competitor>/cannot_reproduce.md` (record attempted config, headline metric achieved, and the delta). Move the competitor to Related-Work prose only. Do NOT retry indefinitely — two configuration sweeps, then STOP.

## §4 — Headline metrics per competitor (filled 2026-04-19)

All metrics below are extracted from the published papers. "10% gate" = `[0.9 × published, 1.1 × published]` — the admissible range for the clean-room reproduction.

| # | Competitor | Original dataset | Headline metric (published) | 10% gate (lower, upper) | Status |
|---|---|---|---|---|---|
| 1 | **LagCNN (Li et al. 2024)** | Weather, 12.5% mask imputation | MSE 0.028, MAE 0.044 | MSE [0.025, 0.031], MAE [0.040, 0.048] | PENDING reproduction |
| 2 | **Wang 2025 CNODE PPMI** | PPMI 5-fold CV | RMSE 0.1606, R² 0.826 | RMSE [0.145, 0.177], R² [0.743, 0.909] | PENDING reproduction |
| 3 | **Demirkaya 2021 EMBC** | Retinal perfusion, SNR 22.56 | CKF MAPE 3.54, NRMSE 0.093 | MAPE [3.19, 3.89], NRMSE [0.084, 0.102] | PENDING reproduction |
| 4 | **Zou 2025 MNODE-HGS** | T1DEXI glucose | RMSE 34.5, Corr 0.68, Diag. Acc. 0.786 | RMSE [31.1, 37.9], Corr [0.61, 0.75], Acc [0.71, 0.86] | PENDING reproduction |

Every row above has its dataset and headline metric confirmed from the published PDF. No "NEEDS PAPER READ" flags remain; Week-1 postdoc research-blocker status is cleared.

**Licensing / action status per competitor:**

- **LagCNN (Li et al. 2024)** — Architecture in paper Eq. 2-13 is fully specified. Clean-room implementable. **No email needed.** Note clearly: LagCNN is a CNN imputation baseline, NOT a physics-regularized competitor. It belongs alongside SAITS / GAIN / MIWAE in the DL-imputer zoo.
- **Wang 2025 CNODE** — arXiv 2511.04789 §II.B-II.D complete. Clean-room implementable. **No email needed.**
- **Demirkaya 2021 EMBC** — Algorithm fully specified in Eq. 4-11. Clean-room implementable. **No email needed to neu-spiral/Hybrid-ODE-NN.**
- **Zou 2025 MNODE-HGS** — arXiv 2505.18996v3 includes full algorithm. Clean-room implementable. **No email needed to bobjz/H2NCM.**
- **de Rooij 2025** — **VENDOR DIRECTLY** from `github.com/Computational-Biology-TUe/ude-regularization` (CC-BY). No clean-room, no email, no fidelity gate.

## §5 — License-request emails (NOT NEEDED)

**No email templates needed — all competitor algorithms are fully specified in their published papers.**

The Week-1 license-request emails previously drafted (Template A to `neu-spiral@northeastern.edu`, Template B to `junyizou@stanford.edu`, Template C to Liang et al. 2024 HSPGNN first author) are **removed from this protocol**. Rationale:

- Template A (Demirkaya 2021): Eq. 4-11 of the EMBC paper fully specifies the Cubature Kalman filter + hybrid ODE-RNN algorithm. No need to request licensed access to `neu-spiral/Hybrid-ODE-NN`.
- Template B (Zou 2025): arXiv 2505.18996v3 supplementary includes the full sparsified hybrid neural ODE algorithm. No need to request licensed access to `bobjz/H2NCM`.
- Template C was drafted for "Liang 2024 HSPGNN" — a paper that does not exist. The DOI 10.1145/3627673.3679672 actually resolves to LagCNN (Li et al. 2024 CIKM), a different paper with different authors. The old template is obsolete.

If a clean-room re-implementation fails Step 4 (>10% gap from published metric) after two sweeps, then — and only then — consider emailing the original authors as a last resort before downgrading the competitor to Related-Work-only citation. The emails are not a Week-1 blocker.

## §6 — Timeline and owner

- **Week 1 — Paper fetch + pseudocode extraction + de Rooij vendoring.** Owner: postdoc Week 1. Deliverable: `spec.md` per clean-room competitor (4 specs: LagCNN, CNODE, Demirkaya 2021, Zou 2025) + vendored de Rooij code under `baselines/derooij2025/`. No license-request emails required.
- **Week 2 — Clean-room implementation.** Owner: postdoc Week 2. Deliverable: runnable code + unit tests under `paper12_phys_gimin/baselines/<name>/`.
- **Week 3 — Original-dataset reproduction + fidelity gate.** Owner: postdoc Week 3. Deliverable: per-competitor PASS / FAIL verdict in `verification_log.md`.

Total elapsed: **3 weeks before the main Paper 12 §V grid can launch**. This aligns with the Paper 12 scoping plan §Revisions (Revision 3) which already carves out 2-3 weeks of verification sprint ahead of the main grid — Task B's §9 timeline confirms the 3-week slot. Any competitor that fails §3 Step 4 is dropped from §V in Week 4.

## §7 — Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Paper does not provide enough detail for re-implementation | LOW | All four competitor papers verified 2026-04-19 to contain full algorithm specifications (LagCNN Eq. 2-13, CNODE §II.B-II.D, Demirkaya Eq. 4-11, Zou algorithm in supplementary). Email the first author only as a last-resort fallback after Step 4 fidelity failure. |
| Original dataset is proprietary / not public (especially CNODE PPMI) | MEDIUM | PPMI access already held by the dissertation project (DUA on file). No separate dataset-access friction. |
| Re-implementation matches original metric within 10% but differs fundamentally in an undocumented way | LOW | Report both numbers in §V + divergence note in the caption |
| Fidelity gate passed on the original dataset but integration with phys-GIMIN's 33-feature schema fails | MEDIUM | Track schema integration as a separate admission checklist item (§8) — do not conflate with fidelity |
| de Rooij vendor code version drifts from published experiments | LOW | Pin `Computational-Biology-TUe/ude-regularization` to a specific git SHA in `baselines/derooij2025/README.md`; record sha256 in run `config.json`. |

All four competitors are clean-room implementable from paper text alone. The de Rooij 2025 competitor is vendorable under CC-BY — no licensing friction in either case. The earlier "license-request lag" risk (R10 in `risk_register.md`) is downgraded from MEDIUM to LOW: emails are no longer a Week-1 blocker since no clean-room reproduction depends on them.

## §8 — Admission checklist

A clean-room baseline enters Paper 12 §V only when every box is ticked.

- [ ] Paper read + pseudocode extracted into `spec.md`
- [ ] Clean-room implementation committed under `paper12_phys_gimin/baselines/<name>/`
- [ ] `LICENSE` (MIT) + `README.md` with independent-re-implementation attribution present
- [ ] Original-dataset reproduction attempted (or explicit "dataset unavailable" note)
- [ ] Headline metric within 10% of published value — OR failure documented in `cannot_reproduce.md`
- [ ] Integration with phys-GIMIN's 33-feature schema complete (adapter in `paper12_phys_gimin/baselines/<name>/adapters.py`)
- [ ] Results recorded in a timestamped run under `paper12_phys_gimin/outputs/runs/<run>/baselines/<name>/`

For **de Rooij 2025** (vendored, not clean-room), the checklist is simpler:

- [ ] `Computational-Biology-TUe/ude-regularization` vendored at pinned git SHA under `paper12_phys_gimin/baselines/derooij2025/`
- [ ] Upstream `LICENSE` (CC-BY) preserved alongside our wrapper `LICENSE` (MIT)
- [ ] Wrapper adapter translates de Rooij's glucose-minimal-model API to phys-GIMIN's 4-feature twin-observable schema (SBR CAUDATE_L/R + PUTAMEN_L/R)
- [ ] `README.md` cites de Rooij et al. 2025 with DOI 10.1371/journal.pcbi.1012198 and links to upstream CC-BY license
- [ ] Integration with phys-GIMIN's 33-feature schema complete
- [ ] Results recorded in a timestamped run under `paper12_phys_gimin/outputs/runs/<run>/baselines/derooij2025/`
