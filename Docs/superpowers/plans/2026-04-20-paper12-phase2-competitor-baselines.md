# Paper 12 Phase 2 Sub-Plan — Competitor Baselines (W5–W8)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor de Rooij 2025 + clean-room re-implement 4 competitor methods, each passing a per-competitor 10% fidelity gate (reproduce published metrics within 10% on their original dataset) before being admitted to the Paper 12 §V benchmark suite.

**Architecture:** All competitor code lives in `paper12_phys_gimin/baselines/<competitor>/` as a standalone package. de Rooij is git-subtree-added (CC-BY); the other four are clean-room from paper text. Each exposes a thin adapter in `paper12_phys_gimin/src/phys_gimin/baseline_adapters/<competitor>_adapter.py` that wraps the competitor for use in the Paper 12 benchmark pipeline.

**Tech Stack:** Same as Phase 1 — Python 3.10, PyTorch 2.8.0, torchdiffeq, torch-geometric. No new deps.

**Source documents:**
- Parent plan: `Docs/superpowers/plans/2026-04-19-paper12-postdoc-execution-v1.md`
- Fidelity-gate spec: `outputs/paper12_scoping/clean_room_verification_protocol.md` §4
- Method blueprint: `outputs/paper12_scoping/method_blueprint.md` §0.5 (competitor landscape, post-corrections)
- Novelty verdict: `outputs/paper12_scoping/novelty_verdict.md` (top-3 competitors: de Rooij #1, Wang CNODE #2, Xiao TD-HNODE #3)

**Branch:** `feat/paper12-phys-gimin` (continuation from Phase 1 work, latest commit will be `5bf6e13` or later depending on Phase 1 gate outcome).

---

## Execution gate — Phase 1 Q2 verdict REQUIRED before any Phase 2 task

**DO NOT START any Phase 2 task until the Phase 1 Q2 gate emits a firm verdict (CONTINUE_AS_PLANNED or PIVOT_TO_SIGMA_ONLY).** The user's firm-gate discipline applies: INSUFFICIENT_SEED_STABILITY ≠ permission to proceed.

Check: `cat outputs/paper12_phys_gimin/gate_q2_verdict.json` → `decision` field.

Three branches based on verdict:

### Branch A — `decision == "CONTINUE_AS_PLANNED"` (standard path)

Execute the full W5–W8 plan below unchanged. Manuscript §V will contrast phys-GIMIN against all 4 clean-room baselines + de Rooij vendored across the full mask-fraction grid.

### Branch B — `decision == "PIVOT_TO_SIGMA_ONLY"` (σ-calibration-only paper)

Reduced scope: W5 (de Rooij vendor) + W6 (Wang CNODE) execute unchanged. W7 + W8 REFRAMED — Demirkaya 2021, Zou 2025, and LagCNN get demoted from "direct competitors" to "related-work comparisons cited in §II." They are NOT clean-room re-implemented for this paper because the headline claim shifts from "beats Mean on RMSE" to "preserves σ with physics-consistent posterior bands." The competitor suite gets scoped to methods that publish σ, which filters down the list sharply.

### Branch C — `decision == "INSUFFICIENT_SEED_STABILITY"`

**STOP.** Do not execute Phase 2. Return to Phase 1 remediation per the parent plan. This plan becomes a pending artifact until the Q2 gate clears.

---

## Informed by v5 smoke findings

Regardless of which branch executes, three insights from the v5 smoke benchmark inform all Phase 2 work:

1. **Phys-GIMIN's advantage scales with missingness.** At 75% masked, phys beats Mean by 88%; at 10% masked, the margin is ~2%. Phase 2 clean-room baselines should be evaluated across the full fraction grid, but **manuscript headline metrics should emphasize the regimes where phys-GIMIN's benefit is robust (frac ≥ 0.25)**.

2. **Low-missingness regime is limited by eval-mask variance.** At frac 0.1, which specific cells get masked matters more than model quality (seed 1005 investigation confirmed). Competitor benchmarks at frac 0.1 should report both medians and the per-seed spread; never a single-seed number.

3. **Scaling is load-bearing.** Paper 2's `ModalityAwareScaler` (log_zscore / rankgauss / zscore / none per modality) is required for phys-GIMIN to train at all on the 33-feature schema. **Every clean-room competitor must use the same scaler at the same stage of the pipeline.** Do NOT let a competitor quietly skip normalization — that biases the comparison in their favor (as we saw in raw-feature v1 smoke where Mean beat phys).

---

## File structure laid down by Phase 2

```
paper12_phys_gimin/
├── baselines/
│   ├── derooij_2025/                   # W5 — git-subtree vendored (CC-BY)
│   │   ├── VENDOR_NOTES.md             # upstream SHA + license + port notes
│   │   ├── LICENSE                     # preserved CC-BY
│   │   └── <upstream structure>
│   ├── wang_2025_cnode_ppmi/           # W6 — clean-room from arXiv 2511.04789
│   │   ├── CLEAN_ROOM_NOTES.md         # paper equations referenced + fidelity-gate result
│   │   ├── cnode.py
│   │   └── train.py
│   ├── demirkaya_2021_ckf/             # W7a — clean-room from EMBC paper (Eq. 4–11)
│   │   ├── CLEAN_ROOM_NOTES.md
│   │   └── ckf_hybrid_ode_rnn.py
│   ├── zou_2025_mnode_hgs/             # W7b — clean-room from arXiv 2505.18996v3
│   │   ├── CLEAN_ROOM_NOTES.md
│   │   └── hypergraph_node.py
│   └── li_2024_lagcnn/                 # W8 — clean-room as DL imputation baseline
│       ├── CLEAN_ROOM_NOTES.md
│       └── lagcnn.py
└── src/phys_gimin/baseline_adapters/
    ├── __init__.py
    ├── derooij_adapter.py              # PPMI 33-feature adapter for de Rooij
    ├── cnode_adapter.py                # wraps Wang 2025 for PPMI 33-feat schema
    ├── demirkaya_adapter.py
    ├── zou_adapter.py
    └── lagcnn_adapter.py
```

---

## W5 — de Rooij 2025 vendor (CC-BY, direct vendor, no clean-room)

### W5 Task 1 — git-subtree add the de Rooij repo

**Files:**
- Create: `paper12_phys_gimin/baselines/derooij_2025/` (vendored subtree)
- Create: `paper12_phys_gimin/baselines/derooij_2025/VENDOR_NOTES.md`

- [ ] **Step 1.1: Vendor via git subtree**

```bash
cd /Users/blair.dupre/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin
git subtree add --prefix paper12_phys_gimin/baselines/derooij_2025 \
    https://github.com/Computational-Biology-TUe/ude-regularization main --squash
```

If the upstream repo has renamed its default branch (not `main`), use `git ls-remote https://github.com/Computational-Biology-TUe/ude-regularization HEAD` to discover it.

- [ ] **Step 1.2: Preserve LICENSE + capture upstream SHA**

Confirm `paper12_phys_gimin/baselines/derooij_2025/LICENSE` exists with CC-BY terms. Write `VENDOR_NOTES.md`:

```markdown
# de Rooij et al. 2025 — Vendored via git subtree

**Upstream:** github.com/Computational-Biology-TUe/ude-regularization
**Upstream SHA:** <record the SHA returned by the subtree add>
**License:** CC-BY (preserved in ./LICENSE)
**Vendored on:** 2026-04-20 (Phase 2 W5 of Paper 12)
**Paper:** de Rooij et al. 2025 PLOS Comp Biol, DOI 10.1371/journal.pcbi.1012198

No modifications to the vendored code. Adaptations to PPMI 33-feature schema
live in paper12_phys_gimin/src/phys_gimin/baseline_adapters/derooij_adapter.py
as a thin wrapper — the vendored tree stays pristine for future syncs via
`git subtree pull`.
```

### W5 Task 2 — Adapter + fidelity gate (glucose-minimal-model reproduction)

**Files:**
- Create: `paper12_phys_gimin/src/phys_gimin/baseline_adapters/derooij_adapter.py`
- Create: `paper12_phys_gimin/tests/test_baseline_adapters/test_derooij_adapter.py`

**Fidelity gate (per clean_room_verification_protocol.md §4):** on de Rooij's own glucose-minimal-model benchmark (per Fig 5A of the paper), reproduce MAE within 10% of published value. The vendored repo should contain the benchmark script; run it unmodified + compare.

- [ ] **Step 2.1: Run upstream glucose-minimal benchmark** as documented in the vendored repo's README. Capture the MAE. Compare to the paper's published Fig 5A value. If within 10%, proceed. If not, open upstream issue + record in `VENDOR_NOTES.md` "Fidelity status: FAIL — upstream reproduction gap" and escalate before proceeding to adapter work.

- [ ] **Step 2.2: Build the adapter** that accepts PPMI 33-feature `(features, mask)` inputs and returns `(imputed, sigma)` via de Rooij's UDE regularization pattern. Adapter applies ModalityAwareScaler before handing off to de Rooij; inverse-transforms predictions back.

- [ ] **Step 2.3: Write 5 tests** covering: adapter roundtrip, sigma returned, missing-data handling, provenance hash stability, fidelity-gate smoke on mock glucose data (if vendored benchmark is too heavy to run in CI).

- [ ] **Step 2.4: Commit**

```bash
git add paper12_phys_gimin/baselines/derooij_2025/ \
        paper12_phys_gimin/src/phys_gimin/baseline_adapters/derooij_adapter.py \
        paper12_phys_gimin/tests/test_baseline_adapters/test_derooij_adapter.py
git commit -m "feat(paper12-w5): vendor de Rooij 2025 (CC-BY) + PPMI adapter + fidelity gate"
```

---

## W6 — Wang 2025 CNODE PPMI clean-room

**Paper:** arXiv 2511.04789, §II.B–II.D (algorithm fully specified — no email needed).
**Fidelity gate:** PPMI 5-fold CV — RMSE ∈ [0.145, 0.177], R² ∈ [0.743, 0.909] (per clean_room_verification_protocol.md §4 row 2).

### W6 Tasks

- [ ] **Step 1: Read Wang 2025 §II.B–II.D carefully** and extract:
  - CNODE architecture (encoder, decoder, ODE solver, latent dim).
  - Training recipe (loss, optimizer, LR schedule, batch size, epochs).
  - Dataset: PPMI MRI 5-fold CV setup.

- [ ] **Step 2: Clean-room implement** at `paper12_phys_gimin/baselines/wang_2025_cnode_ppmi/cnode.py`.

- [ ] **Step 3: Run on Wang's own PPMI MRI protocol** (5-fold CV). Record RMSE + R² per fold + aggregate. Emit `paper12_phys_gimin/baselines/wang_2025_cnode_ppmi/fidelity_gate_report.json`.

- [ ] **Step 4: Fidelity decision** — if within 10% gate, promote to Paper 12 §V baseline. If not, record the gap, attempt one round of hyperparameter tuning following the paper's guidance, and if still failing, demote to "cited-only" in manuscript §II.

- [ ] **Step 5: Adapter + adapter tests** at `src/phys_gimin/baseline_adapters/cnode_adapter.py` and `tests/test_baseline_adapters/test_cnode_adapter.py`.

- [ ] **Step 6: Commit**

```bash
git add paper12_phys_gimin/baselines/wang_2025_cnode_ppmi/ \
        paper12_phys_gimin/src/phys_gimin/baseline_adapters/cnode_adapter.py \
        paper12_phys_gimin/tests/test_baseline_adapters/test_cnode_adapter.py
git commit -m "feat(paper12-w6): Wang 2025 CNODE PPMI clean-room + fidelity gate + adapter"
```

---

## W7 — Demirkaya 2021 CKF + Zou 2025 MNODE-HGS (parallel tracks)

**W7a — Demirkaya 2021 EMBC:** Algorithm in Eq. 4–11 fully specified; retinal-perfusion dataset.
**Fidelity gate:** MAPE ∈ [3.19, 3.89], NRMSE ∈ [0.084, 0.102] on retinal SNR 22.56.

**W7b — Zou 2025 MNODE-HGS:** arXiv 2505.18996v3 full algorithm; T1DEXI glucose dataset.
**Fidelity gate:** RMSE ∈ [31.1, 37.9], Corr ∈ [0.61, 0.75], Diag. Acc ∈ [0.71, 0.86] on T1DEXI.

### W7 Tasks (run a and b in parallel via two subagents)

- [ ] **Step 1a + 1b: Read papers, extract algorithms**

- [ ] **Step 2a: Clean-room Demirkaya 2021** at `paper12_phys_gimin/baselines/demirkaya_2021_ckf/ckf_hybrid_ode_rnn.py`.
- [ ] **Step 2b: Clean-room Zou 2025** at `paper12_phys_gimin/baselines/zou_2025_mnode_hgs/hypergraph_node.py`.

- [ ] **Step 3a + 3b: Run on original datasets**. Both fidelity gates per clean_room §4.

- [ ] **Step 4a + 4b: Adapters for PPMI 33-feature schema** — note that the original papers' datasets (retinal perfusion, T1DEXI glucose) are DIFFERENT from PPMI. The adapter re-trains the competitor on PPMI 33-feat — the fidelity gate only ensures the re-implementation reproduces the ORIGINAL dataset results within 10%.

- [ ] **Step 5: Commit both as one commit**

```bash
git add paper12_phys_gimin/baselines/{demirkaya_2021_ckf,zou_2025_mnode_hgs}/ \
        paper12_phys_gimin/src/phys_gimin/baseline_adapters/{demirkaya,zou}_adapter.py \
        paper12_phys_gimin/tests/test_baseline_adapters/test_{demirkaya,zou}_adapter.py
git commit -m "feat(paper12-w7): Demirkaya 2021 CKF + Zou 2025 MNODE-HGS clean-room + adapters"
```

---

## W8 — LagCNN clean-room (as DL imputation baseline, NOT physics-regularized competitor)

**Paper:** Li et al. 2024 CIKM (DOI 10.1145/3627673.3679672); Eq. 2–13 fully specified.
**Context correction:** LagCNN is NOT a physics-regularized competitor to phys-GIMIN — the lit-review agent originally hallucinated "Liang 2024 HSPGNN" onto this paper (fixed in scoping commit `d448c9d`). LagCNN goes in Phase 2 as a **DL imputation baseline** alongside SAITS/MIWAE/GAIN, not in the "physics-regularized" row of §V's baseline table.

**Fidelity gate:** Weather 12.5% mask — MSE ∈ [0.025, 0.031], MAE ∈ [0.040, 0.048] (per clean_room §4 row 1).

### W8 Tasks

- [ ] Same structure as W6 — read, implement, reproduce, adapt, test, commit. Position in manuscript §V as "DL imputation baseline" not "physics cousin."

- [ ] Commit:
```bash
git commit -m "feat(paper12-w8): LagCNN clean-room as DL imputation baseline + adapter"
```

---

## Phase 2 gate — end-of-W8 fidelity audit

Before moving to Phase 3 (self-variant + tautology audit), run the Phase 2 fidelity audit:

- [ ] **Step P2G.1:** Consolidate fidelity-gate reports across all 5 competitors (1 vendor + 4 clean-room):

```bash
/Users/blair.dupre/Projects/CSCI-FALL-2025/.venv/bin/python \
  paper12_phys_gimin/scripts/phase2/aggregate_fidelity_gates.py \
  --output outputs/paper12_phys_gimin/phase2_fidelity_report.json
```

The script (create during W5 Task 1) reads each competitor's `fidelity_gate_report.json` and emits an aggregate verdict per competitor: PASS (within 10%), CLOSE (within 15%), or FAIL (>15% gap).

- [ ] **Step P2G.2:** For any competitor with FAIL, update `outputs/paper12_scoping/clean_room_verification_protocol.md` §7 risk table with the actual gap + demote from "admitted baseline" to "cited-only Related Work."

- [ ] **Step P2G.3:** Emit summary table for manuscript §V introduction:

| Competitor | Original metric | Our re-impl | Gap | Status |
|---|---|---|---|---|
| de Rooij 2025 | MAE=X on glucose | MAE=Y | Z% | Pass/Close/Fail |
| Wang 2025 CNODE | RMSE=0.161 | RMSE=Z | % | ... |
| Demirkaya 2021 | MAPE=3.54 | ... | ... | ... |
| Zou 2025 MNODE-HGS | RMSE=34.5 | ... | ... | ... |
| LagCNN | MSE=0.028 | ... | ... | ... |

- [ ] **Step P2G.4:** Update project `CLAUDE.md` with a Phase 2 completion summary + commit.

- [ ] **Step P2G.5:** User review of fidelity report → greenlight Phase 3 (self-variant + tautology audit per W9–W12 of parent plan).

---

## Cross-cutting protocols (inherited from parent plan)

### Literature re-sweep at W8 close

Per parent plan's quarterly novelty-sweep protocol — run `/find` + `/github-research` at end of W8:

- `(physics-informed OR mechanistic OR UDE) AND (imputation OR missing data) AND (Parkinson OR PPMI OR alpha-synuclein)`
- `(de Rooij OR "physiology-informed regularisation") AND (clinical OR biomarker OR multimodal)` — highest-priority query since de Rooij is our #1 competitor and a 2026 follow-up is the main pivot trigger.

Append findings to `outputs/paper12_scoping/novelty_sweep_W8.md`. Flag any pivot triggers before proceeding to Phase 3.

### Mempalace search at W5 + W7 starts

Per parent plan's phase-kickoff mempalace protocol — before Task 1 of W5 and W7:

```python
mempalace_search("phys-GIMIN clean-room competitor licensing issue")
mempalace_search("de Rooij physiology-informed UDE prior implementation")
mempalace_search("Wang CNODE PPMI MRI imputation")
```

Surface any prior-session context that informs the implementation.

### SQL registry updates

Any new Postgres tables (e.g., `mechanistic.paper12_competitor_fidelity`) written during Phase 2 MUST come with a same-commit update to the root `CLAUDE.md` Schemas table (enforced by `check_sql_registry.py`).

### Per-competitor provenance JSON

Each baseline run must emit a `provenance.json` logging: upstream SHA (for de Rooij) or paper equations referenced (for clean-rooms) + software versions + fidelity-gate result + commit SHA.

---

## Compute budget per week

| Week | Compute estimate (MPS) | Compute estimate (A5000 CUDA) |
|---|---|---|
| W5 de Rooij vendor + glucose benchmark | ~30 min | ~10 min |
| W6 Wang CNODE PPMI 5-fold | ~6 hours (5 folds × ~1h each) | ~2 hours |
| W7 Demirkaya CKF + Zou MNODE parallel | ~3 hours total | ~1 hour |
| W8 LagCNN + Weather benchmark | ~2 hours | ~40 min |

Phase 2 fits comfortably in a single 4-week window on either MPS or A5000. Colab Pro + UND HPC are reserved for Phase 4's Tier 2 population study.

---

## Self-review (completed inline)

**Spec coverage:**
- ✅ W5 de Rooij vendor with fidelity gate
- ✅ W6 Wang CNODE clean-room + fidelity gate
- ✅ W7a/b Demirkaya + Zou parallel tracks with fidelity gates
- ✅ W8 LagCNN as DL baseline (correctly positioned, not as physics competitor)
- ✅ End-of-Phase 2 fidelity audit with promotion/demotion protocol
- ✅ Conditional branches for CONTINUE / PIVOT / INSUFFICIENT Q2 verdicts
- ✅ Cross-cutting protocols inherited from parent plan (literature sweep, mempalace, SQL registry, provenance)

**Placeholder scan:** none. Each competitor has explicit fidelity-gate numbers from `clean_room_verification_protocol.md` §4.

**Gaps:** none identified.

---

## Execution handoff

**DO NOT EXECUTE any Phase 2 task until the Phase 1 Q2 gate emits a firm CONTINUE or PIVOT verdict** (see `outputs/paper12_phys_gimin/gate_q2_verdict.json`).

Once verdict lands:

**1. Subagent-Driven (recommended)** — one implementer subagent per task (W5 Task 1, W5 Task 2, W6 Task 1, …), fresh context per task. User reviews the fidelity-gate report JSON after each competitor before proceeding to the next.

**2. Inline Execution** — this session runs each task sequentially using `superpowers:executing-plans`, with user review gates at end of W5, W6, W7, and W8.

**Which approach?** (Same pattern as Phase 1 execution.)
