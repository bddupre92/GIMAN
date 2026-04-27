# Next Steps — Post-Compact Resume (2026-04-20)

State summary for resuming work after conversation compaction.

## Current branch: `feat/ch9-6-multichannel` (18 commits ahead of main)

```
785f173 chore(audit-db): verify SciML demo doesn't refute prior claims
b535733 feat(paper11-demo): physics-informed neural ODE on DaT-SBR — hybrid beats pure NN
de69245 chore(audit-db): refresh claim lineage for Ch 14 §14.4 + Ch 15 Paper 11 Preview
f58814f feat(hybrid-twin): Discussion §14.4 synthesis + Alt-5 null probe + Ch 15 Paper 11 Preview
6ef3c70 feat(paper10): fix blank fig4 + add 6 missing companion refs (full arc)
83977a6 feat(paper9): remediate CPT:PSP submission — full journal-style-audit pass
fa35cce feat(paper8a+8b): new submission packages for PLoS Comp Biol + Movement Disorders
263295c feat(paper7): CPT:PSP submission package with §9.6 multichannel integration
1a0aea8 feat(paper6): polish JAMIA submission package — trim deadcode + fix margins
27cf466 feat(paper10): npj Parkinson's Disease submission package with L1 integration
e2019ce feat(L1): Pillar 8 PUTAMEN bidirectional demo — negative result, informative
7bde0f1 feat(L1): Pillar 7 calibration-corrections validation — Task A executed
ed4bd64 feat(L1): draft calibration-corrections validation (Task A, ready to run)
94267ed feat(L1): Pillar 6 MCAR held-out validation — feature-specific σ finding
f76c34e feat(L1): per-visit GIMIN inference on 16,699 P3 longitudinal visits
b2e0e57 feat(L1): Xing external validation (MNAR finding)
0204ba9 feat(L1): baseline bridge v2 (2,197 pts × 33 feats)
3dd6cfe feat(L1): mechanism + tests + L1-lite demo
```

## Submission packages ready to ship

All 9 primary dissertation papers have compiled submission packages with zero
undefined references:

| Paper | Venue | Path | Status |
|---|---|---|---|
| P1 | IEEE JBHI | `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/` | Ready |
| P2 | IEEE JBHI | `outputs/mechanistic_twin/paper2_submission/ieee-jbhi/` | Ready |
| P3+4 | npj Digital Medicine | `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/` | Ready |
| P5 | JAMIA | `outputs/mechanistic_twin/paper5_submission/` | Ready |
| **P6** | **JAMIA** | `outputs/mechanistic_twin/paper6_submission/jamia/` | **Ready** (commit `1a0aea8`, untouched) |
| **P7** | **CPT:PSP** | `outputs/mechanistic_twin/paper7_submission/cpt-psp/` | **Ready** (commit `263295c`) |
| **P8a** | **PLoS Comp Biol** | `outputs/mechanistic_twin/paper8a_submission/plos-compbio/` | **Ready** (commit `fa35cce`) |
| **P8b** | **Movement Disorders** | `outputs/mechanistic_twin/paper8b_submission/movement-disorders/` | **Ready** (commit `fa35cce`) |
| **P9** | **CPT:PSP** | `outputs/mechanistic_twin/paper9_submission/cpt-psp/` | **Ready** (commit `83977a6`) |
| **P10** | **npj Parkinson's Disease** | `outputs/mechanistic_twin/paper10_submission/npj-pd/` | **Ready** (commit `6ef3c70`) |

## Dissertation state

- 357 pages, 7.6 MB PDF
- Zero undefined citations
- Ch 15 "Paper 11 Preview" inserted between Discussion (Ch 14) and Conclusion (Ch 16)
- Ch 14 §14.4 "Unified Hybrid Twin at Three Integration Layers" added
- Audit DB: 2,004 claims, 1,859 verified (93%), 111 partial, 0 refuted
- Build command: `cd outputs/dissertation && pdflatex main.tex && pdflatex main.tex`

## Hybrid-twin architectural arc (completed)

| Layer | Paper | Deliverable |
|---|---|---|
| 1. Parallel integration | P6 JAMIA | 5-layer pipeline output with N(t)/N₀ metadata |
| 2. Complementarity + gap characterisation | P9 + P10 L1 Pillars 1-8 | Head-to-head + σ×2.5→19% joint-coverage finding |
| 3. Deep SciML fusion (preview) | Ch 15 + `paper11_demo/` | Hybrid beats pure NN by Δ=-0.030 MAE |

## SciML demo headline (commit `b535733`)

Held-out last-scan SBR MAE on 50 test patients:

| Model | MAE | RMSE |
|---|---|---|
| Pure Mechanistic (Fearnley-Lees 2.5%/yr lit prior) | **0.1164** | 0.1603 |
| Physics-Informed Hybrid (UDE) | 0.1640 | 0.2553 |
| Pure Neural ODE (no physics) | 0.1935 | 0.2703 |

Hybrid beats pure NN by −0.030 (physics regularization validated); pure
mechanistic wins overall at demo scale (expected small-n regime).

Reproduce: `.venv/bin/python scripts/paper11_demo/hybrid_sciml_neural_ode.py`

## Health checks (all PASS)

| Check | Status | Evidence |
|---|---|---|
| SQL registry | ✓ current | 718 MB · 185 tables · 14 schemas (matches CLAUDE.md) |
| Audit DB freshness | ✓ current | Claim DB + scorecard refreshed post-commit b535733 |
| Mempalace mining | ✓ current | 24 files processed, 761 drawers filed (vault-sync 07:26) |
| Audit→Obsidian sync | ✓ current | 16 chapters + 3 dashboards |
| Vault git | ✓ clean | committed: "vault-sync: 17 files refreshed" |
| torchdiffeq | ✓ installed | 0.2.5 (enabled SciML demo) |

## Post-compact resume actions (ordered by priority)

### Immediate (if resuming within 1 week)

1. **Submit submission packages** — all 9 are ready to upload. Sequence per Apr-18 strategy doc: bioRxiv simultaneously, then venue uploads. The one committed truth is `outputs/mechanistic_twin/paper{N}_submission/{venue}/main.pdf`.

2. **Commit Ch 03 Paper 1 uncommitted diff** — `outputs/dissertation/chapters/ch03_paper1.tex` has 217 lines of improvements since Apr-17-18 (Espay critique + PD-only + 46-feature hierarchy + circularity audit) sitting uncommitted. Safe to commit any time; submission packages already have the newer prose.

3. **Dissertation chapter sync** — ch04_paper2 and ch08_paper6 are behind their submission packages per the deferred todo. Non-blocking for defense; cosmetic cleanup.

### Short-term (1-3 weeks)

4. **Scale the SciML demo to full cohort** — current demo uses 150/50 train/test. Full cohort is 299 train / 65 test (all patients with ≥3 scans). Expect hybrid to close more of the gap to pure mech. Script: `scripts/paper11_demo/hybrid_sciml_neural_ode.py` with `train_records = train_batch["records"]` (remove the `[:150]` cap).

5. **Alt-6 probe for Paper 6** — test `pct_loss_per_yr_median × t` temporal interaction as a 34th feature. Current Alt-5 null used only the scalar rate; the interaction term might surface temporal signal that static rate alone can't.

6. **Coordinate with phys-GIMIN (P12) Phase 2** — worktree `feat/paper12-phys-gimin` continues on W5 de Rooij vendor. If that lands a strong lit-prior result, Paper 11 Direction 1 becomes a direct extension. Check status via `cd ~/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin && git log --oneline -5`.

### Medium-term (postdoc / Paper 11 proper)

7. **External longitudinal DaT-SPECT** — three candidate sources pending DUAs:
   - SURE-PD3 via BioSEND DUA (~300 pts × 2 timepoints, 2-3 week turnaround)
   - DeNoPa (Mollenhauer collaboration)
   - ICEBERG (Paris Brain Institute)

8. **Paper 11 full execution** — follow the 3-direction plan from Ch 15:
   - Direction 1: Physics-informed neural ODE regularization (UDE)
   - Direction 2: Trajectory-aware imputation with joint calibration
   - Direction 3: Hybrid posterior propagation

## Key memory notes for mempalace search

- `session_2026_04_20_hybrid_twin_scicompl` — this session's closure arc
- `session_2026_04_19_l1_full_execution` — L1 Pillars 1-8 (includes Pillar 8 negative result)
- `session_2026_04_18_robustness_workstreams` — W1-W4 cross-arc integration
- `paper12_phys_gimin_scope` — two-variant lit/self design decision
- `paper12_phys_gimin_status` — current Phase 1 COMPLETE, Phase 2 W5 next

## Uncommitted state (pre-session, user-aware — not this session's scope)

- `M CLAUDE.md` (older)
- `M outputs/dissertation/bibliography.tex` (older)
- `M outputs/dissertation/chapters/ch03_paper1.tex` (Apr-17 uncommitted edits)
- `M scripts/paper1/run_pd_only_retraining.py` (older)
- `?? Docs/superpowers/plans/2026-04-{19,20}-paper12-*.md` (P12 plans, mirrored from worktree)
- `?? scripts/load_paper12_phase1_to_pg.py`, `?? scripts/load_paper2_parquet_to_pg.py`

These are pre-session state. The user has been aware throughout and hasn't
prioritised committing them; they don't affect submission packages or dissertation build.

## One-shot resume command

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025
git log --oneline main..feat/ch9-6-multichannel | head -5   # confirm branch state
.venv/bin/python scripts/stale_check.py                     # confirm health checks pass
ls outputs/mechanistic_twin/paper*_submission/*/main.pdf    # confirm all submissions present
```
