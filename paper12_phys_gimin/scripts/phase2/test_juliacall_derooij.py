"""Smoke test: load de Rooij's Julia regularizer functions from Python via juliacall.

Usage
-----
    python paper12_phys_gimin/scripts/phase2/test_juliacall_derooij.py

Expected output (abbreviated):
    Julia version: 1.12.x
    Bridge env: .../julia_bridge_env/
    Bridge functions available: derooij_nonneg_loss, derooij_auc_loss
    nonneg_loss([1, -0.5, 2, -0.3, 0]) = 0.34  (expected: 0.34)
    auc_loss(uniform 1/480 x 481 pts) = ~0.0    (expected: 0.0 ± 1e-14)
    SMOKE TEST PASSED

Notes
-----
juliacall picks up whichever Julia is default on PATH (1.12.x).  The bridge env
at ``julia_bridge_env/`` pins only Trapz v2 and is compat-declared for Julia
1.10, 1.11, 1.12 — so the regularizer functions work on any of these versions.

De Rooij's full UDE environment (``baselines/derooij_2025/``) requires Julia
1.10.0 or 1.10.4 (per their Manifest.toml) and also needs DifferentialEquations +
Lux to precompile.  We do NOT instantiate that environment here — the regularizer
functions only need Trapz, which is why the bridge env exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve project root relative to this script
_SCRIPT_DIR = Path(__file__).parent
_PKG_ROOT = _SCRIPT_DIR.parent.parent  # paper12_phys_gimin/
BRIDGE_ENV = _PKG_ROOT / "src" / "phys_gimin" / "baseline_adapters" / "julia_bridge_env"
BRIDGE_FILE = _PKG_ROOT / "src" / "phys_gimin" / "baseline_adapters" / "derooij_bridge.jl"


def main() -> None:
    try:
        from juliacall import Main as jl
    except ImportError:
        print("ERROR: juliacall not installed. Run: pip install juliacall", file=sys.stderr)
        sys.exit(1)

    print(f"Julia version: {jl.VERSION}")
    print(f"Bridge env:    {BRIDGE_ENV}")

    # Activate the minimal Trapz-only environment
    jl.seval(f'import Pkg; Pkg.activate("{BRIDGE_ENV}")')
    # Load the bridge functions
    jl.include(str(BRIDGE_FILE))

    # Confirm functions are defined
    names = [str(n) for n in jl.names(jl.Main) if not str(n).startswith("#")]
    print(f"Bridge functions available: {[n for n in names if 'derooij' in n.lower()]}")

    # Test 1: nonneg_loss
    import numpy as np
    ra1 = np.array([1.0, -0.5, 2.0, -0.3, 0.0])
    nonneg = float(jl.derooij_nonneg_loss(ra1))
    expected_nonneg = (-0.5)**2 + (-0.3)**2  # 0.34
    print(f"\nnonneg_loss([1, -0.5, 2, -0.3, 0]) = {nonneg:.6f}  (expected: {expected_nonneg:.6f})")
    assert abs(nonneg - expected_nonneg) < 1e-12, f"nonneg mismatch: got {nonneg}, expected {expected_nonneg}"

    # Test 2: auc_loss with uniform rate that integrates to 1 over [0, 480]
    ra2 = np.full(481, 1.0 / 480.0)
    times2 = np.arange(481, dtype=np.float64)
    auc = float(jl.derooij_auc_loss(ra2, times2))
    print(f"auc_loss(uniform 1/480 x 481 pts) = {auc:.2e}  (expected: ~0.0)")
    assert abs(auc) < 1e-12, f"auc_loss should be ~0, got {auc}"

    # Test 3: auc_loss with non-unit integral
    ra3 = np.full(481, 2.0 / 480.0)  # integrates to 2, so penalty = |2 - 1| = 1
    auc3 = float(jl.derooij_auc_loss(ra3, times2))
    print(f"auc_loss(uniform 2/480 x 481 pts) = {auc3:.6f}  (expected: 1.0)")
    assert abs(auc3 - 1.0) < 1e-10, f"auc_loss should be 1.0, got {auc3}"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
