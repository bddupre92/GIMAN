# Julia Setup Log — de Rooij 2025 Bridge Integration

## Summary

Phase 2 W5 juliacall integration completed 2026-04-19.  
Julia 1.10.x installed via juliaup.  
juliacall 0.9.31 operational with Julia 1.12.6 (see version note below).  
Numerical equivalence verified: Python regularizers match Julia to < 1e-10.

---

## Step 1: Julia 1.10.x Installation

**Method:** juliaup (already installed at `~/.juliaup/bin/juliaup`)

```bash
juliaup add 1.10
julia +1.10 --version  # → julia version 1.10.11
```

**Result:** Julia 1.10.11 installed at:
```
~/.julia/juliaup/julia-1.10.11+0.aarch64.apple.darwin14/
```

**Version note:** De Rooij's `Project.toml` declares `julia = "1.10.0, 1.10.4"` in the
`[compat]` section.  This constraint applies to their FULL UDE training pipeline (which
requires `DifferentialEquations`, `Lux`, `SciMLSensitivity`).  The bridge we use
**only requires Trapz** — the constraint is irrelevant.  We use **Julia 1.12.6** (the
juliaup default), which juliacall links at startup.  Julia 1.10.11 is installed for
reference and for any future full-pipeline runs.

---

## Step 2: De Rooij Full Environment — Partial Instantiation

```bash
cd baselines/derooij_2025
julia +1.10 --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Result:** 737 packages precompiled successfully; DifferentialEquations FAILED.

**Error:**
```
UndefVarError: `default_linear_interpolation` not defined
 @ OrdinaryDiffEqSymplecticRK
```

**Root cause:** The Manifest.toml was pinned for Julia 1.10.0/1.10.4 exactly.  Julia
1.10.11's updated OrdinaryDiffEq packages renamed internal functions.  This affects
only the ODE solver — not the regularizer functions we need.

**Decision:** Defer full instantiation. The regularizer functions in `ude.jl` only use
`Trapz` (and base Julia `sum`, `min`, `abs`).  We created a minimal bridge environment.

---

## Step 3: Minimal Bridge Environment

Created `src/phys_gimin/baseline_adapters/julia_bridge_env/` with only `Trapz` as
a dependency.

```bash
julia --project=julia_bridge_env -e 'using Pkg; Pkg.instantiate(); using Trapz; println(pkgversion(Trapz))'
# → 2.0.3  (OK)
```

**Trapz version:** 2.0.3 (matches de Rooij's `[compat]` constraint `Trapz = "2"`).

---

## Step 4: juliacall Installation

```bash
pip install juliacall
# → juliacall-0.9.31 installed
```

**Julia version used by juliacall:** 1.12.6 (default juliaup channel).  
Setting `PYTHON_JULIACALL_BINDIR` to Julia 1.10 fails because `juliacall` already
linked `libjulia.1.12.dylib` from the default `julia` binary on first import.
Julia 1.12.6 is fully compatible with Trapz 2.0.3 and all bridge functions.

---

## Step 5: Smoke Test

```bash
python paper12_phys_gimin/scripts/phase2/test_juliacall_derooij.py
```

**Output:**
```
Julia version: 1.12.6
Bridge env: .../julia_bridge_env
Bridge functions available: ['derooij_auc_loss', 'derooij_nonneg_loss']
nonneg_loss([1, -0.5, 2, -0.3, 0]) = 0.340000  (expected: 0.340000)
auc_loss(uniform 1/480 x 481 pts) = 2.78e-15  (expected: ~0.0)
auc_loss(uniform 2/480 x 481 pts) = 1.000000  (expected: 1.0)
SMOKE TEST PASSED
```

---

## Step 6: Numerical Equivalence Tests

10 tests in `tests/test_baseline_adapters/test_derooij_julia_equivalence.py`.  
All 10 PASS.  Max absolute difference over 20-sweep random arrays:

| Regularizer | Max |Python - Julia| |
|-------------|----------------------|
| nonneg      | < 1e-10              |
| auc         | < 1e-10              |

**Note on float precision:** Python helpers use float64 (`.double()` in torch) to match
Julia's default Float64 arithmetic.  This achieves < 1e-10 across all tests.  Downstream
PPMI imputation uses float32, which naturally introduces ~1e-6 differences — well within
the stated fidelity gate.

---

## Full Test Suite

```
94 passed, 804 warnings in 13.30s
```

(84 existing + 10 new equivalence tests)

---

## Files Created

| File | Purpose |
|------|---------|
| `src/.../julia_bridge_env/Project.toml` | Minimal Trapz-only Julia env |
| `src/.../julia_bridge_env/Manifest.toml` | Auto-generated lock |
| `src/.../derooij_bridge.jl` | Bridge: top-level Julia functions for juliacall |
| `tests/test_baseline_adapters/test_derooij_julia_equivalence.py` | Fidelity gate tests |
| `scripts/phase2/test_juliacall_derooij.py` | Smoke test script |
| `baselines/derooij_2025/JULIA_SETUP_LOG.md` | This file |
