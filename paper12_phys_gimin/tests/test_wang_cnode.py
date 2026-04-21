"""Unit tests for the Wang 2025 CNODE clean-room baseline + phys-GIMIN adapter.

Five tests covering:

1. ``test_cnode_forward_shape``
       forward() returns (batch, T, feature_dim).
2. ``test_cnode_train_step_reduces_loss``
       a single Adam step on a synthetic batch reduces the MSE loss.
3. ``test_adapter_respects_33_feature_schema``
       CnodeAdapter.fit/impute handle PPMI (N, 33) tensors without crashing.
4. ``test_fidelity_gate_bounds_are_locked``
       FIDELITY_GATE_RMSE_BOUNDS == (0.145, 0.177) and R² bounds exactly match
       the pre-registered protocol. Regression test against silent drift.
5. ``test_synthetic_mode_produces_fidelity_report``
       train.py --synthetic writes report.json with the expected schema.

Run:

    ./.venv/bin/python -m pytest paper12_phys_gimin/tests/test_wang_cnode.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add the baselines directory to sys.path so we can import the clean-room package
_BASELINE_DIR = (
    Path(__file__).resolve().parent.parent / "baselines"
).resolve()
if str(_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_DIR))


# ---------------------------------------------------------------------------
# Test 1: forward pass shape
# ---------------------------------------------------------------------------

def test_cnode_forward_shape() -> None:
    """CNODE.forward() returns (batch, T, feature_dim)."""
    pytest.importorskip("torch")
    pytest.importorskip("torchdiffeq")
    import torch

    from wang_2025_cnode_ppmi.cnode import CNODE, CNODEConfig

    torch.manual_seed(0)
    config = CNODEConfig(feature_dim=12, covariate_dim=4, hidden_dim=8)
    model = CNODE(config)

    batch = 3
    x_0 = torch.randn(batch, config.feature_dim)
    c = torch.randn(batch, config.covariate_dim)
    t_eval = torch.tensor([0.0, 0.5, 1.0, 2.0])

    out = model(x_0, c, t_eval)
    assert out.shape == (batch, t_eval.numel(), config.feature_dim), (
        f"Expected ({batch}, {t_eval.numel()}, {config.feature_dim}), got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "Output must be finite."


# ---------------------------------------------------------------------------
# Test 2: one training step reduces loss
# ---------------------------------------------------------------------------

def test_cnode_train_step_reduces_loss() -> None:
    """A single Adam step on a synthetic batch strictly reduces trajectory MSE."""
    pytest.importorskip("torch")
    pytest.importorskip("torchdiffeq")
    import torch

    from wang_2025_cnode_ppmi.cnode import CNODE, CNODEConfig

    torch.manual_seed(42)
    config = CNODEConfig(feature_dim=6, covariate_dim=3, hidden_dim=8)
    model = CNODE(config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    batch, T = 4, 3
    x_0 = torch.randn(batch, config.feature_dim)
    c = torch.randn(batch, config.covariate_dim)
    t_eval = torch.tensor([0.0, 1.0, 2.0])
    target = torch.randn(batch, T, config.feature_dim)
    mask = torch.ones(batch, T)

    # Loss before
    pred_0 = model(x_0, c, t_eval)
    loss_0 = CNODE.trajectory_mse(pred_0, target, mask).item()

    # One step
    opt.zero_grad()
    pred = model(x_0, c, t_eval)
    loss = CNODE.trajectory_mse(pred, target, mask)
    loss.backward()
    opt.step()

    # Loss after
    pred_1 = model(x_0, c, t_eval)
    loss_1 = CNODE.trajectory_mse(pred_1, target, mask).item()

    assert loss_1 < loss_0, f"Expected loss decrease, got {loss_0:.6f} -> {loss_1:.6f}"


# ---------------------------------------------------------------------------
# Test 3: CnodeAdapter respects the 33-feature schema
# ---------------------------------------------------------------------------

def test_adapter_respects_33_feature_schema() -> None:
    """CnodeAdapter.fit()/impute() handle (N, 33) inputs and return matching shapes."""
    pytest.importorskip("torch")
    from phys_gimin.baseline_adapters import CnodeAdapter

    rng = np.random.default_rng(0)
    n, d = 20, 33
    features = rng.uniform(0.5, 10.0, size=(n, d)).astype(np.float32)
    mask = (rng.uniform(size=(n, d)) > 0.2).astype(np.float32)
    # Ensure at least one observed entry per column
    for j in range(d):
        if mask[:, j].sum() == 0:
            mask[0, j] = 1.0

    adapter = CnodeAdapter(device="cpu", sigma_fallback=0.1, seed=42)
    adapter.fit(features, mask)
    mu, sigma = adapter.impute(features, mask)

    assert mu.shape == (n, d), f"mu shape expected ({n}, {d}), got {mu.shape}"
    assert sigma.shape == (n, d), f"sigma shape expected ({n}, {d}), got {sigma.shape}"
    assert np.all(np.isfinite(mu)), "mu must be finite"
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0), "sigma must be positive & finite"

    # Observed entries are left unchanged
    obs_r, obs_c = np.where(mask == 1)
    np.testing.assert_array_almost_equal(
        mu[obs_r, obs_c],
        features[obs_r, obs_c],
        decimal=5,
        err_msg="Observed entries must pass through unchanged.",
    )


# ---------------------------------------------------------------------------
# Test 4: fidelity gate bounds are LOCKED
# ---------------------------------------------------------------------------

def test_fidelity_gate_bounds_are_locked() -> None:
    """Regression test: the pre-registered fidelity gate bounds must not drift.

    Source of truth:
        outputs/paper12_scoping/clean_room_verification_protocol.md §4 row 2
        (Wang 2025 CNODE PPMI: RMSE 0.1606, R² 0.826, ±10% gate).

    The bounds enumerated here must match ``FIDELITY_GATE_RMSE_BOUNDS`` and
    ``FIDELITY_GATE_R2_BOUNDS`` in ``train.py`` exactly. A silent change to
    those module-level constants — even to "match what we actually produce" —
    will fail this test. This is the intended safety net against moving the
    goalposts after we see the real-data numbers.
    """
    from wang_2025_cnode_ppmi.train import (
        FIDELITY_GATE_R2_BOUNDS,
        FIDELITY_GATE_RMSE_BOUNDS,
    )

    assert FIDELITY_GATE_RMSE_BOUNDS == (0.145, 0.177), (
        f"RMSE bounds drifted to {FIDELITY_GATE_RMSE_BOUNDS}; "
        "update clean_room_verification_protocol.md §4 before changing this."
    )
    assert FIDELITY_GATE_R2_BOUNDS == (0.743, 0.909), (
        f"R² bounds drifted to {FIDELITY_GATE_R2_BOUNDS}; "
        "update clean_room_verification_protocol.md §4 before changing this."
    )


# ---------------------------------------------------------------------------
# Test 5: synthetic mode writes a fidelity report with the expected schema
# ---------------------------------------------------------------------------

def test_synthetic_mode_produces_fidelity_report(tmp_path: Path) -> None:
    """train.py --synthetic runs end-to-end and writes a well-formed report.json.

    This verifies the harness itself — NOT fidelity to Wang's published metrics.
    Per CLEAN_ROOM_NOTES.md §5, synthetic-feature eval is a smoke test only.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchdiffeq")

    from wang_2025_cnode_ppmi.cnode import CNODEConfig
    from wang_2025_cnode_ppmi.train import generate_synthetic_cohort, run

    # Keep the smoke run small: 20 patients, 3 epochs per fold, 2 folds.
    cohort = generate_synthetic_cohort(
        n_patients=20,
        feature_dim=12,
        covariate_dim=3,
        seed=7,
    )
    config = CNODEConfig(feature_dim=12, covariate_dim=3, hidden_dim=8)

    out_dir = tmp_path / "cnode_smoke"
    report = run(
        cohort=cohort,
        config=config,
        out_dir=out_dir,
        n_folds=2,
        n_epochs=3,
        lr=1e-2,
        seed=7,
        device="cpu",
        mode="synthetic",
        verbose=False,
    )

    # Files were written
    assert (out_dir / "report.json").exists(), "report.json must be written"
    assert (out_dir / "config.json").exists(), "config.json must be written"

    # Report schema — the downstream benchmark harness relies on these keys
    for key in (
        "paper",
        "clean_room",
        "mode",
        "n_patients",
        "n_folds",
        "fidelity_gate",
        "per_fold",
        "runtime_seconds",
    ):
        assert key in report, f"report.json missing required key '{key}'"

    gate = report["fidelity_gate"]
    for key in (
        "rmse_bounds",
        "r2_bounds",
        "mean_rmse",
        "std_rmse",
        "mean_r2",
        "std_r2",
        "rmse_pass",
        "r2_pass",
        "verdict",
        "note",
    ):
        assert key in gate, f"fidelity_gate missing required key '{key}'"

    assert gate["verdict"] in {"PASS", "FAIL"}, (
        f"verdict must be PASS or FAIL, got {gate['verdict']!r}"
    )
    assert report["mode"] == "synthetic"
    assert len(report["per_fold"]) == 2
    # Each per-fold entry has the right keys
    for m in report["per_fold"]:
        for k in ("fold", "rmse", "r2", "n_train", "n_test"):
            assert k in m, f"per_fold entry missing '{k}'"

    # Re-read the file to confirm it's valid JSON
    with open(out_dir / "report.json") as f:
        roundtrip = json.load(f)
    assert roundtrip["fidelity_gate"]["verdict"] == gate["verdict"]
