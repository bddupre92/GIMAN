"""Smoke benchmark runner — importable + CLI-wrapped.

Functions in this module are importable for unit-testing and also invoked
by scripts/phase1/run_smoke_benchmark.py as CLI entry points.

Pipeline per seed:
  1. Load PPMI 33-feature dataset (real, or a mock dataset fixture for tests).
  2. Apply MCAR mask at specified mask_fraction using the seed RNG.
  3. Train phys-GIMIN-lit for n_epochs via PhysGIMINTrainer.
  4. Compute RMSE on masked-but-observed entries.
  5. Run Mean baseline for reference (fills missing with column means).
  6. Emit per-seed JSON: outputs/paper12_phys_gimin/runs/{name}_seed{N}_{ts}/results.json.

Multi-seed + multi-fraction orchestration is done by run_multi_seed() which
dispatches single-seed runs in sequence.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phys_gimin.model import PhysGIMIN
from phys_gimin.priors.literature import LiteraturePriorProvider
from phys_gimin.regularizer import PhysicsRegularizer
from phys_gimin.trajectory_cache import TrajectoryCache
from phys_gimin.training import PhysGIMINTrainer


@dataclass
class SmokeRunResult:
    method: str                   # "phys_gimin_lit" or "mean"
    seed: int
    mask_fraction: float
    final_rmse: float
    wall_time_s: float
    run_dir: str
    n_epochs: int
    n_patients: int
    n_features: int
    status: str                   # "completed" or "failed"


# ── Tiny mock dataset (replicates _TinyDataset from test_training.py) ─────────

class _MockDataset(Dataset):
    """Mock dataset for smoke benchmark unit tests.

    n_patients × n_features synthetic data.
    ode_feature_indices=[0, 1] — first 2 features treated as DaT-SBR.
    t_years has 2 elements to match len(ode_feature_indices).
    """

    def __init__(self, n_patients: int = 50, n_features: int = 33, seed: int = 1001):
        rng = np.random.default_rng(seed)
        self.features = torch.from_numpy(rng.standard_normal((n_patients, n_features))).float()
        self.mask = torch.ones(n_patients, n_features)
        self.stage_ids = torch.from_numpy(rng.integers(0, 6, (n_patients,))).long()
        self.patnos = list(range(3000, 3000 + n_patients))
        # t_years has 2 elements — matches len(ode_feature_indices)=2
        self.t_years = [np.array([0.0, 1.0])] * n_patients
        self.sbr_0 = [2.5] * n_patients
        self.n_patients = n_patients
        self.n_features = n_features

    def __len__(self) -> int:
        return 1  # one full-graph batch

    def __getitem__(self, idx: int) -> dict:
        n = self.n_patients
        # build a minimal connected edge list: chain 0→1→2→...→n-1 (undirected)
        srcs = list(range(n - 1)) + list(range(1, n))
        dsts = list(range(1, n)) + list(range(n - 1))
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_weight = torch.ones(len(srcs))
        overlap_frac = torch.ones(len(srcs))
        return {
            "features": self.features,
            "mask": self.mask,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "overlap_frac": overlap_frac,
            "stage_ids": self.stage_ids,
            "patnos": self.patnos,
            "t_years_per_patient": self.t_years,
            "sbr_0_per_patient": self.sbr_0,
            "ode_feature_indices": [0, 1],
        }


def _collate_passthrough(batch: list[dict]) -> dict:
    """batch is a length-1 list of dicts; return the dict directly."""
    return batch[0]


def _apply_mcar_mask(features: torch.Tensor, mask_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MCAR mask to features. Returns (masked_features, mask).

    mask[i, j] = 1 means observed; mask[i, j] = 0 means missing.
    masked_features has 0.0 at missing positions.
    """
    rng = np.random.default_rng(seed)
    n, f = features.shape
    mask = torch.ones(n, f)
    # For each entry independently, set to missing with probability mask_fraction
    missing_indicator = rng.random((n, f)) < mask_fraction
    mask[missing_indicator] = 0.0
    masked_features = features.clone()
    masked_features[mask == 0] = 0.0
    return masked_features, mask


def _compute_rmse(predicted: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute RMSE on originally-masked entries (mask==0 positions)."""
    missing = mask == 0
    if missing.sum() == 0:
        # No masked entries — evaluate on all
        diff = predicted - true
    else:
        diff = predicted[missing] - true[missing]
    return float(torch.sqrt((diff ** 2).mean()).item())


def _run_mean_baseline(
    true_features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fill missing entries with per-column mean from observed entries."""
    n, f = true_features.shape
    filled = true_features.clone()
    for j in range(f):
        col_obs_mask = mask[:, j] == 1
        if col_obs_mask.any():
            col_mean = true_features[col_obs_mask, j].mean()
        else:
            col_mean = torch.tensor(0.0)
        missing_rows = mask[:, j] == 0
        filled[missing_rows, j] = col_mean
    return filled


def run_single_seed(
    method: str,
    seed: int,
    mask_fraction: float,
    n_epochs: int,
    n_patients: int,
    n_features: int,
    output_dir: Path,
    mock_data: bool = True,
) -> SmokeRunResult:
    """Run one seed at one mask fraction. Returns result + writes JSON to output_dir."""
    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{method}_seed{seed}_{ts}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Build data
        if mock_data:
            dataset = _MockDataset(n_patients=n_patients, n_features=n_features, seed=seed)
            true_features = dataset.features  # shape (n_patients, n_features)
        else:
            raise NotImplementedError(
                "Real PPMI data loading not implemented in unit-test path. "
                "Pass mock_data=True for tests, or implement DB-backed loading here."
            )

        # 2. Apply MCAR mask
        masked_features, mask = _apply_mcar_mask(true_features, mask_fraction, seed=seed + 10000)

        if method == "mean":
            # 3+4. Mean baseline: fill with column means on observed data, compute RMSE
            predicted = _run_mean_baseline(masked_features, mask)
            final_rmse = _compute_rmse(predicted, true_features, mask)
            actual_epochs = 1

        elif method == "phys_gimin_lit":
            # 3. Build PhysGIMIN + trainer
            torch.manual_seed(seed)
            model = PhysGIMIN(
                modality_dims=[2, 5, 6, 6, 4, 4, 6],
                embed_dim=64,
                num_gnn_layers=2,
                num_heads=4,
                mc_dropout=0.1,
                num_stages=6,
                stage_embed_dim=16,
            )
            provider = LiteraturePriorProvider()
            regularizer = PhysicsRegularizer(provider=provider, beta=0.5)
            cache = TrajectoryCache(provider=provider)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = PhysGIMINTrainer(
                model=model,
                regularizer=regularizer,
                trajectory_cache=cache,
                optimizer=optimizer,
                seed=seed,
                warmup_epochs=1,
                lambda_phys_target=1.0,
                min_recon_fraction=0.30,
                output_dir=run_dir,
                run_name="phys_gimin_lit",
                deterministic=True,
                device="cpu",
            )

            # Build dataset with masked features
            train_dataset = _MockDataset(n_patients=n_patients, n_features=n_features, seed=seed)
            # Override features with masked version and mask
            train_dataset.features = masked_features
            train_dataset.mask = mask

            loader = DataLoader(train_dataset, batch_size=1, collate_fn=_collate_passthrough)

            # 4. Train
            train_result = trainer.fit(loader, loader, n_epochs=n_epochs, patience=n_epochs + 10)

            if train_result.status == "failed":
                raise RuntimeError(f"Training failed: {train_result.failure_reason}")

            # 5. Compute RMSE on masked entries
            model.eval()
            full_dataset = _MockDataset(n_patients=n_patients, n_features=n_features, seed=seed)
            full_dataset.features = masked_features
            full_dataset.mask = mask
            full_loader = DataLoader(full_dataset, batch_size=1, collate_fn=_collate_passthrough)

            with torch.no_grad():
                batch = next(iter(full_loader))
                output = model(
                    batch["features"],
                    batch["mask"],
                    batch["edge_index"],
                    batch["edge_weight"],
                    batch["overlap_frac"],
                    batch["stage_ids"],
                )
            predicted = output["imputed_mean"]
            final_rmse = _compute_rmse(predicted, true_features, mask)
            actual_epochs = train_result.final_epoch + 1

        else:
            raise ValueError(f"Unknown method: {method!r}. Must be 'phys_gimin_lit' or 'mean'.")

        wall_time_s = time.time() - t0
        status = "completed"

    except Exception as exc:
        wall_time_s = time.time() - t0
        status = "failed"
        final_rmse = float("nan")
        actual_epochs = 0
        # Write error info
        (run_dir / "error.txt").write_text(str(exc))

    result = SmokeRunResult(
        method=method,
        seed=seed,
        mask_fraction=mask_fraction,
        final_rmse=final_rmse,
        wall_time_s=wall_time_s,
        run_dir=str(run_dir),
        n_epochs=actual_epochs,
        n_patients=n_patients,
        n_features=n_features,
        status=status,
    )

    # Write per-seed JSON
    (run_dir / "results.json").write_text(json.dumps(asdict(result), indent=2))
    return result


def run_multi_seed(
    method: str,
    seeds: list[int],
    mask_fraction: float,
    n_epochs: int,
    n_patients: int = 50,
    n_features: int = 33,
    output_dir: Path = Path("outputs/paper12_phys_gimin/runs/smoke_test"),
    mock_data: bool = True,
) -> list[SmokeRunResult]:
    """Run multiple seeds sequentially; return list of results."""
    results = []
    for seed in seeds:
        result = run_single_seed(
            method=method,
            seed=seed,
            mask_fraction=mask_fraction,
            n_epochs=n_epochs,
            n_patients=n_patients,
            n_features=n_features,
            output_dir=output_dir,
            mock_data=mock_data,
        )
        results.append(result)
    return results


def rmse_cv(results: list[SmokeRunResult]) -> float:
    """Coefficient of variation of final_rmse across a list of seed results."""
    rmses = np.array([r.final_rmse for r in results if r.status == "completed"])
    if len(rmses) == 0 or np.mean(rmses) == 0:
        return float("inf")
    return float(np.std(rmses) / np.mean(rmses))
