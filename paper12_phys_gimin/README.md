# paper12_phys_gimin

Physics-regularized multimodal imputation for Parkinson's disease biomarker data
(Paper 12, postdoc scope).

## Status

**Week 1 of 16-week postdoc execution.** Scoping phase complete —
see `outputs/paper12_scoping/` in the main repo for the 13 scoping deliverables.
Scoping plan at `~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md`.

**Scholar-eval verdict (2026-04-19):** READY for postdoc execution with
composite 6.4/10 (PASS-WITH-REVISIONS). All 3 mandatory revisions from D7
self-review are addressed in the scoping plan.

## Hard constraints

This directory is **standalone**. It:

- **Imports** from `giman_pipeline.imputation`, `giman_pipeline.mechanistic_twin_v2`,
  `giman_pipeline.paper3`, `giman_pipeline.paper4`.
- **Reads** existing output files (HDF5 posteriors, Paper 2 checkpoints,
  `PerFeatureTemperatureScaler` state dicts).
- **Subclasses / composes** existing classes — never forks them.
- **Does NOT modify** any file in the main project's `src/giman_pipeline/`,
  `scripts/`, or `GIMImpN_imputation/`.

See scoping plan "Standalone-directory architecture" section for rationale.

## Method at a glance

Two variants sharing one architecture:

- **phys-GIMIN-lit:** physics regularizer uses literature-anchored ODE parameters
  (Fearnley-Lees N₀, Lee 2019 γ=0.7, Iljina 2016 rate constants). Zero leakage.
- **phys-GIMIN-self:** physics regularizer uses per-patient posteriors from
  `giman_pipeline.mechanistic_twin_v2.posterior_store`. Partially tautological
  against Papers 7/9/10 — evaluated only on Paper 2 + Paper 3 downstream,
  with explicit ⚠ tautology flag in results tables.

Both use β-NLL loss (Seitzer 2022, β=0.5) with **stop-gradient on σ inside
L_physics** (prevents explain-away of aleatoric variance).

## Layout

```
paper12_phys_gimin/
├── README.md                     (this file)
├── pyproject.toml                Minimal deps (inherits main-project .venv)
├── LICENSE                       MIT
├── .gitignore                    outputs/, caches
├── src/phys_gimin/
│   ├── priors/
│   │   ├── base.py               PriorProvider Protocol
│   │   ├── literature.py         Lit-anchored ODE constants
│   │   └── posterior_store.py    Reads main-project HDF5
│   ├── regularizer.py            PhysicsRegularizer (β-NLL + stop-grad)
│   ├── loss.py                   MVP 2-term loss
│   ├── model.py                  PhysGIMIN (Week 2+ — stubbed)
│   ├── training.py               LR-annealing loop (Week 3+ — stubbed)
│   └── observation_adapters/     SBR, multichannel, Path B (Week 5+)
├── scripts/                      Benchmark + downstream runners (Week 8+)
├── baselines/                    Clean-room competitor re-implementations (Week 1-3)
├── configs/                      YAML per variant (Week 1)
├── tests/                        pytest — Week 1 scope: priors + stop_grad
└── outputs/                      Gitignored; benchmark runs
```

## Dev quick-start

```bash
# Use the main project venv (shared)
source /Users/blair.dupre/Projects/CSCI-FALL-2025/.venv/bin/activate

# Run tests
cd ~/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin
pytest paper12_phys_gimin/tests/ -v
```

## Key references

- Scoping plan: `~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md`
- Method blueprint: `outputs/paper12_scoping/method_blueprint.md` (18 concepts)
- Best practices: `outputs/paper12_scoping/impl_best_practices.md`
- Clean-room protocol: `outputs/paper12_scoping/clean_room_verification_protocol.md`
- Experiment plans: `outputs/paper12_scoping/experiment_plan_{lit,self}.md`
- Risk register: `outputs/paper12_scoping/risk_register.md`
- Target venue: npj Systems Biology and Applications (APC $3,290, OA)
- Preprint deadline: 2026 Q4 (9-15 month freshness window per D3)

## Session log

- **2026-04-19 Week 1 kickoff:** worktree + scaffold + PriorProvider interface
  + PhysicsRegularizer skeleton + 2 load-bearing unit tests. This commit.
