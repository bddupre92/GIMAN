"""5-fold CV training harness + pre-registered fidelity gate for Wang 2025 CNODE.

Pre-registered fidelity gate (LOCKED — do NOT modify without protocol amendment):

    RMSE 5-fold CV mean ∈ [0.145, 0.177]
    R²   5-fold CV mean ∈ [0.743, 0.909]

Source: outputs/paper12_scoping/clean_room_verification_protocol.md §4 row 2
(published RMSE 0.1606, R² 0.826 ± 10% window). These bounds are locked by
``tests/test_wang_cnode.py::test_fidelity_gate_bounds_are_locked`` — any future
refactor that silently alters them will fail the test.

Usage
-----
Synthetic smoke test (no FreeSurfer required):

    python -m wang_2025_cnode_ppmi.train --synthetic \\
        --out outputs/runs/cnode_smoke

Real FreeSurfer features (when available at data/02_freesurfer/):

    python -m wang_2025_cnode_ppmi.train --real \\
        --feat-dir data/02_freesurfer \\
        --out outputs/runs/cnode_real

The synthetic mode generates random feature matrices of the correct shape
(68 subcortical + 148 vertex-thickness) and DOES NOT constitute a fidelity
attestation — it is a smoke test of the training loop and output JSON schema.
A true fidelity verdict requires the ``--real`` mode once FreeSurfer outputs are
available.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam

# Support both installed (`pip install -e .`) and in-repo invocation.
try:
    from wang_2025_cnode_ppmi.cnode import CNODE, CNODEConfig
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wang_2025_cnode_ppmi.cnode import CNODE, CNODEConfig


# ---------------------------------------------------------------------------
# Pre-registered fidelity gate bounds (LOCKED)
# ---------------------------------------------------------------------------

FIDELITY_GATE_RMSE_BOUNDS: tuple[float, float] = (0.145, 0.177)
FIDELITY_GATE_R2_BOUNDS: tuple[float, float] = (0.743, 0.909)

# Wang 2025 cohort (per scoping doc §4 row 2):
#   N=161 patients: 111 with 2 visits, 50 with ≥3 visits
WANG_2025_N_PATIENTS: int = 161
WANG_2025_N_2VISITS: int = 111
WANG_2025_N_3VISITS: int = 50
WANG_2025_FEATURE_DIM: int = 216   # 68 subcortical vols + 148 vertex-wise thickness
WANG_2025_COVARIATE_DIM: int = 8


# ---------------------------------------------------------------------------
# Synthetic data generator (smoke test — NOT a fidelity attestation)
# ---------------------------------------------------------------------------

def generate_synthetic_cohort(
    n_patients: int = WANG_2025_N_PATIENTS,
    feature_dim: int = WANG_2025_FEATURE_DIM,
    covariate_dim: int = WANG_2025_COVARIATE_DIM,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return a dict of random tensors approximating Wang's PPMI shape.

    Trajectories are generated from a linear drift + small noise so that an
    *idealised* CNODE could fit them perfectly. The synthetic signal is a
    placeholder for genuine morphometry; see CLEAN_ROOM_NOTES.md §5 for the
    distinction between smoke-test and fidelity-test.
    """
    rng = np.random.default_rng(seed)

    x_0 = rng.normal(size=(n_patients, feature_dim)).astype(np.float32)
    c = rng.normal(size=(n_patients, covariate_dim)).astype(np.float32)

    # Build a (n_patients, T=4) time grid with observed mask.
    # Real cohort: 111 have 2 visits, 50 have ≥3 visits. We sample T=4 grid
    # [0, 1, 2, 4] years; mask visits 3-4 for the "2-visit" subset.
    t_eval = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    mask = np.ones((n_patients, len(t_eval)), dtype=np.float32)
    idx_2visit = rng.permutation(n_patients)[: WANG_2025_N_2VISITS]
    mask[idx_2visit, 2:] = 0.0

    # Synthetic trajectory: small per-patient drift direction modulated by c.
    drift = rng.normal(scale=0.05, size=(n_patients, feature_dim)).astype(np.float32)
    drift += 0.02 * (c[:, :1] * rng.normal(size=(n_patients, feature_dim))).astype(np.float32)

    trajectories = np.stack(
        [x_0 + drift * t_k for t_k in t_eval], axis=1
    ).astype(np.float32)  # (N, T, D)

    # Add a small amount of observation noise.
    trajectories = trajectories + rng.normal(scale=0.01, size=trajectories.shape).astype(np.float32)

    return {
        "x_0": x_0,
        "c": c,
        "t_eval": t_eval,
        "trajectories": trajectories,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stratify_folds(mask: np.ndarray, n_folds: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Stratified K-fold by visit count (2-visit vs 3+-visit subgroups).

    Mirrors Wang's cohort split: preserves the 111/50 ratio within each fold so
    the test set is never devoid of long trajectories.
    """
    rng = np.random.default_rng(seed)
    n = mask.shape[0]
    visit_counts = mask.sum(axis=1)                      # (N,)
    is_long = (visit_counts >= 3).astype(np.int64)       # 0 = 2-visit, 1 = ≥3-visit

    folds: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_folds)]
    for stratum in (0, 1):
        idx = np.where(is_long == stratum)[0]
        rng.shuffle(idx)
        splits = np.array_split(idx, n_folds)
        for k in range(n_folds):
            folds[k] = np.concatenate([folds[k], splits[k]])
    # Deterministic sort within each fold for stable JSON output.
    return [np.sort(f) for f in folds]


def _r_squared(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Masked R² averaged over observed (sample, t, feature) entries."""
    m = mask[..., None].astype(np.float32)
    obs = target * m
    pred_m = pred * m
    # Residual sum of squares
    ss_res = np.sum((obs - pred_m) ** 2)
    # Total sum of squares around the observed mean
    n_obs = m.sum() * pred.shape[-1]
    mean_obs = obs.sum() / (n_obs + 1e-8)
    ss_tot = np.sum(((obs - mean_obs * m) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = mask[..., None].astype(np.float32)
    diff = (pred - target) * m
    n_obs = m.sum() * pred.shape[-1]
    return float(math.sqrt((diff ** 2).sum() / (n_obs + 1e-8)))


def train_fold(
    cohort: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: CNODEConfig,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, float]:
    """Train CNODE on train_idx, evaluate on test_idx. Returns metrics dict."""
    x_0_all = torch.tensor(cohort["x_0"], dtype=torch.float32, device=device)
    c_all = torch.tensor(cohort["c"], dtype=torch.float32, device=device)
    traj_all = torch.tensor(cohort["trajectories"], dtype=torch.float32, device=device)
    mask_all = torch.tensor(cohort["mask"], dtype=torch.float32, device=device)
    t_eval = torch.tensor(cohort["t_eval"], dtype=torch.float32, device=device)

    model = CNODE(config).to(device)
    opt = Adam(model.parameters(), lr=lr)

    train_x0 = x_0_all[train_idx]
    train_c = c_all[train_idx]
    train_traj = traj_all[train_idx]
    train_mask = mask_all[train_idx]

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        pred = model(train_x0, train_c, t_eval)
        loss = CNODE.trajectory_mse(pred, train_traj, train_mask)
        loss.backward()
        opt.step()
        if verbose and (epoch % 25 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d} | train MSE = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_pred = model(x_0_all[test_idx], c_all[test_idx], t_eval).cpu().numpy()
    test_traj = traj_all[test_idx].cpu().numpy()
    test_mask = mask_all[test_idx].cpu().numpy()

    return {
        "rmse": _rmse(test_pred, test_traj, test_mask),
        "r2": _r_squared(test_pred, test_traj, test_mask),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def run(
    cohort: dict[str, np.ndarray],
    config: CNODEConfig,
    out_dir: Path,
    n_folds: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    mode: str = "synthetic",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run 5-fold stratified CV and write a fidelity gate JSON report.

    Parameters
    ----------
    cohort : dict of ndarrays with keys {x_0, c, t_eval, trajectories, mask}.
    config : CNODEConfig hyperparameter container.
    out_dir : destination for report.json + config.json.
    n_folds : number of CV folds (default 5, locked by fidelity protocol).
    n_epochs : per-fold training epochs.
    lr : Adam learning rate.
    seed : RNG seed applied to torch / numpy / random.
    device : PyTorch device string.
    mode : "synthetic" or "real". Stamped into the report for downstream filtering.
    verbose : per-epoch logging.
    """
    _set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = _stratify_folds(cohort["mask"], n_folds=n_folds, seed=seed)
    n = cohort["x_0"].shape[0]
    all_idx = np.arange(n)

    t0 = time.time()
    per_fold: list[dict[str, float]] = []
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.setdiff1d(all_idx, test_idx)
        fold_metrics = train_fold(
            cohort=cohort,
            train_idx=train_idx,
            test_idx=test_idx,
            config=config,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
        )
        fold_metrics["fold"] = k
        per_fold.append(fold_metrics)
        if verbose:
            print(
                f"fold {k}: rmse={fold_metrics['rmse']:.4f} "
                f"r2={fold_metrics['r2']:.4f}"
            )

    mean_rmse = float(np.mean([m["rmse"] for m in per_fold]))
    mean_r2 = float(np.mean([m["r2"] for m in per_fold]))
    std_rmse = float(np.std([m["rmse"] for m in per_fold], ddof=1))
    std_r2 = float(np.std([m["r2"] for m in per_fold], ddof=1))

    rmse_low, rmse_high = FIDELITY_GATE_RMSE_BOUNDS
    r2_low, r2_high = FIDELITY_GATE_R2_BOUNDS
    rmse_pass = rmse_low <= mean_rmse <= rmse_high
    r2_pass = r2_low <= mean_r2 <= r2_high
    gate_pass = bool(rmse_pass and r2_pass)

    report: dict[str, Any] = {
        "paper": "Wang et al. 2025 CNODE (arXiv:2511.04789)",
        "clean_room": True,
        "mode": mode,
        "n_patients": int(n),
        "n_folds": n_folds,
        "n_epochs_per_fold": n_epochs,
        "seed": seed,
        "device": device,
        "fidelity_gate": {
            "rmse_bounds": list(FIDELITY_GATE_RMSE_BOUNDS),
            "r2_bounds": list(FIDELITY_GATE_R2_BOUNDS),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "rmse_pass": rmse_pass,
            "r2_pass": r2_pass,
            "verdict": "PASS" if gate_pass else "FAIL",
            "note": (
                "Synthetic-feature evaluation is a smoke test of the training loop "
                "and output schema — NOT a fidelity attestation. A true verdict "
                "requires --real with FreeSurfer outputs from data/02_freesurfer/."
                if mode == "synthetic"
                else "Real-feature 5-fold CV evaluation."
            ),
        },
        "per_fold": per_fold,
        "runtime_seconds": float(time.time() - t0),
    }

    # Write report.json + config.json (config.json is readable by train.py reruns)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": asdict(config),
                "training": {
                    "n_folds": n_folds,
                    "n_epochs": n_epochs,
                    "lr": lr,
                    "seed": seed,
                    "device": device,
                    "mode": mode,
                },
            },
            f,
            indent=2,
        )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wang 2025 CNODE clean-room fidelity gate harness."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on synthetic features (smoke test, NOT a fidelity attestation).",
    )
    mode.add_argument(
        "--real",
        action="store_true",
        help="Run on real FreeSurfer features at --feat-dir.",
    )
    p.add_argument(
        "--feat-dir",
        type=Path,
        default=Path("data/02_freesurfer"),
        help="Directory with real FreeSurfer outputs (used with --real).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/runs/wang_cnode"),
        help="Output directory for report.json + config.json.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _load_real_cohort(feat_dir: Path) -> dict[str, np.ndarray]:
    """Load real FreeSurfer cohort from disk.

    Expects ``feat_dir/cohort.npz`` with keys ``x_0``, ``c``, ``t_eval``,
    ``trajectories``, ``mask``. If the file does not exist a descriptive error
    is raised so the user knows exactly what to prepare.
    """
    npz_path = feat_dir / "cohort.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Real cohort not found at {npz_path}. "
            "FreeSurfer preprocessing has not produced a cohort.npz yet. "
            "Run --synthetic first to smoke-test the harness."
        )
    data = np.load(npz_path)
    expected = {"x_0", "c", "t_eval", "trajectories", "mask"}
    missing = expected - set(data.files)
    if missing:
        raise KeyError(f"cohort.npz is missing keys: {sorted(missing)}")
    return {k: data[k] for k in expected}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.synthetic:
        cohort = generate_synthetic_cohort(seed=args.seed)
        mode = "synthetic"
    else:
        cohort = _load_real_cohort(args.feat_dir)
        mode = "real"

    config = CNODEConfig(
        feature_dim=cohort["x_0"].shape[1],
        covariate_dim=cohort["c"].shape[1],
    )
    report = run(
        cohort=cohort,
        config=config,
        out_dir=args.out,
        n_folds=args.n_folds,
        n_epochs=args.n_epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        mode=mode,
        verbose=args.verbose,
    )
    verdict = report["fidelity_gate"]["verdict"]
    print(f"Fidelity gate: {verdict}  |  rmse={report['fidelity_gate']['mean_rmse']:.4f}  "
          f"r2={report['fidelity_gate']['mean_r2']:.4f}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
