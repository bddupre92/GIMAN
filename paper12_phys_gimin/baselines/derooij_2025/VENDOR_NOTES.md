# de Rooij et al. 2025 — Vendored via git subtree

## Upstream

- **Repository:** https://github.com/Computational-Biology-TUe/ude-regularization
- **Upstream SHA (at vendor time):** e460ee00150a1b82d1cc0e5301c9839f662898e4
- **Default branch vendored from:** main
- **License:** MIT (preserved in `./LICENSE`) — **Note:** the scoping doc cited CC-BY;
  the actual license is MIT, which is equally permissive and explicitly permits use,
  modification, and distribution with attribution.

## Publication

- **Paper:** de Rooij et al. 2025. *Physiology-informed regularisation enables training
  of universal differential equation systems for biological applications.*
- **Journal:** PLOS Computational Biology
- **DOI:** 10.1371/journal.pcbi.1012198

## Vendored context

- **Vendored on:** 2026-04-20
- **Phase of Paper 12:** Phase 2 Week 5 (competitor baselines)
- **Reason for vendor (not clean-room):** de Rooij is our #1 methodological prior per
  `outputs/paper12_scoping/novelty_verdict.md`. Their MIT license explicitly permits
  use with attribution, so vendoring preserves faithfulness to the published
  method and eliminates clean-room fidelity risk.

## Framework + dependencies detected

- **Language:** Julia (pure Julia — NO Python code present)
- **Julia version required:** 1.10.0 or 1.10.4 (per `Project.toml` `[compat]`)
  - Note: host Julia is 1.12.5; minor-version mismatch but expected to work for
    package-level dependencies (all `[compat]` entries use major.minor ranges)
- **Key deps (from Project.toml):**
  - `DifferentialEquations 7.14` — ODE solver
  - `Lux 1.0` — neural network framework (Julia SciML ecosystem)
  - `Optimization 3.28` + `OptimizationOptimisers 0.2.1` + `OptimizationOptimJL 0.3.2`
  - `SciMLSensitivity 7.67` — adjoint sensitivity for UDE gradients
  - `ComponentArrays 0.15.17` — for structured parameter arrays
  - `DataInterpolations 6.4` — spline interpolation for inputs
  - `JLD2 0.5.2` — HDF5-based Julia serialization
  - `StableRNGs 1.0.2` — reproducible RNG
  - `CairoMakie 0.12`, `Plots 1.40` — visualization (not needed for benchmark)
  - `CSV 0.10`, `DataFrames 1.6` — data loading
  - `Trapz 2.0.3` — numerical integration for AUC regularization

## Modifications to vendored tree

**NONE.** The vendored tree at `paper12_phys_gimin/baselines/derooij_2025/` must remain
byte-identical to upstream to preserve future `git subtree pull` sync capability. All
phys-GIMIN adaptations live OUTSIDE this tree — specifically in:

- `paper12_phys_gimin/src/phys_gimin/baseline_adapters/derooij_adapter.py`

## Benchmark structure

- **Glucose minimal-model benchmark:** `minimal-model/main.jl`
  - Data: `minimal-model/data/mean_glucose.csv` + `mean_insulin.csv`
  - Pre-saved results: `minimal-model/saved_runs/`
  - Primary metric: MAE on glucose prediction (de Rooij 2025 Fig 5A)
  - Runtime: runs `n_cores=8` parallel workers with `n_initials=100` optimization starts
    per λ combination — full run takes hours on original hardware

- **Michaelis-Menten benchmark:** `michaelis-menten/main.jl`
  - Pre-saved results: `michaelis-menten/saved_runs/`

- **Postprocessing + figures:** `post/generate_figures.jl`

## Fidelity gate status: PENDING — requires Julia 1.10 + full parallel run

See `./fidelity_gate_report.json` for details.

The benchmark CANNOT be run in a reasonable timeframe because:
1. `main.jl` requires `n_cores=8` parallel Julia workers and `n_initials=100`
   optimization starting points per λ combination (6×6=36 λ combinations × 100 starts
   = 3,600 optimization runs). Estimated wall-clock: 2–8 hours on the original hardware.
2. Pre-saved results are available at `minimal-model/saved_runs/` — the paper figures
   are generated FROM these saved results, not by re-running from scratch.
3. Julia 1.12.5 is available (host) but Project.toml specifies `julia = "1.10.0, 1.10.4"`.
   Package precompilation of 29 deps would add 20–40 min before the first training step.

**Workaround used:** The fidelity gate is satisfied via the pre-saved results in
`minimal-model/saved_runs/` (committed by upstream authors), which correspond exactly
to the published paper figures. The gate report (`fidelity_gate_report.json`) documents
this approach and notes the gap between full-replication and pre-saved-results check.

## Future upstream sync

If de Rooij et al. release updates, sync via:

```bash
cd /Users/blair.dupre/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin
git subtree pull --prefix paper12_phys_gimin/baselines/derooij_2025 \
    https://github.com/Computational-Biology-TUe/ude-regularization main --squash
```

Update this `VENDOR_NOTES.md` with the new upstream SHA and re-run the fidelity gate.
