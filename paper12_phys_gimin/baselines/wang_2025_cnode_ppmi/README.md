# Wang 2025 CNODE — clean-room re-implementation

Independently re-implemented from:

> Wang, X., Zhao, Y., Han, K., Luo, X., van Rooij, S., Stevens, J., He, L.,
> Zhan, L., Sun, Y., Wang, W., Yang, C. (2025).
> "Conditional Neural ODE for Longitudinal Parkinson's Disease Progression
> Forecasting." arXiv:2511.04789.

**No source was consulted from any upstream repository.** Every architectural and
training choice is documented in [`CLEAN_ROOM_NOTES.md`](./CLEAN_ROOM_NOTES.md),
including every place where the paper under-specifies and a standard default
was used.

## Purpose

Paper 12 (phys-GIMIN) Phase 2 Week 6 baseline competitor. The CNODE is a pure
data-driven Neural ODE for PD morphometry progression — a relevant benchmark
because phys-GIMIN's novelty claim is specifically about *physics-regularized*
imputation, which assumes data-driven NODEs like CNODE are the appropriate
ablation.

## Pre-registered fidelity gate (LOCKED)

Per `outputs/paper12_scoping/clean_room_verification_protocol.md` §4 row 2:

| Metric | Wang 2025 published | Admissible clean-room range (±10%) |
|---|---|---|
| RMSE | 0.1606 | **[0.145, 0.177]** |
| R²   | 0.826  | **[0.743, 0.909]** |

These bounds are enforced by
`tests/test_wang_cnode.py::test_fidelity_gate_bounds_are_locked`.

The gate is evaluated on **real** FreeSurfer features (N=161 PPMI patients,
111×2-visit + 50×≥3-visit). The `--synthetic` mode is a **smoke test only** —
see §5 of CLEAN_ROOM_NOTES.md.

## Quick start

```bash
# Smoke test (no FreeSurfer required — synthetic features)
python -m wang_2025_cnode_ppmi.train --synthetic \
    --out outputs/runs/cnode_smoke --verbose

# Real evaluation (requires data/02_freesurfer/cohort.npz)
python -m wang_2025_cnode_ppmi.train --real \
    --feat-dir data/02_freesurfer \
    --out outputs/runs/cnode_real --verbose
```

The harness writes `report.json` + `config.json` to the output directory.
`report.json` contains the full fidelity-gate verdict, per-fold metrics, and a
clear stamp distinguishing smoke-test from fidelity-test mode.

## Files

- `cnode.py` — `CNODE` model + `CNODEConfig` dataclass.
- `train.py` — 5-fold stratified-CV harness with locked fidelity bounds.
- `CLEAN_ROOM_NOTES.md` — documented assumptions.
- `pyproject.toml` — minimal deps (torch, torchdiffeq, numpy).
- `LICENSE` — MIT.

## Adapter for phys-GIMIN benchmark

See `src/phys_gimin/baseline_adapters/cnode_adapter.py` for the wrapper that
accepts the PPMI 33-feature schema and routes only brain-imaging channels to
CNODE while falling back to feature-mean imputation for non-imaging channels.
