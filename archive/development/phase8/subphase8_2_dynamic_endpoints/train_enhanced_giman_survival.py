"""Minimal survival trainer for Phase 8.2 enhanced cohort.

Trains a small MLP on node features saved in the PyG Data objects
produced by `prepare_enhanced_training_data.py`. Uses the Cox
partial-likelihood loss (Breslow approximation) and reports
concordance index on validation and test splits.

This script is intentionally small and self-contained so it can be
used as a smoke-test before integrating with the full GIMAN trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import argparse
import math

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data


def load_data(path: Path) -> Data:
    """Load a single PyG Data object from a file.

    Args:
        path: Path to the saved .pt file.

    Returns:
        A torch_geometric.data.Data instance.
    """
    obj = torch.load(path)
    if not isinstance(obj, Data):
        raise TypeError("Expected a PyG Data object in the file")
    return obj


class MLPRisk(nn.Module):
    """Simple MLP that outputs a scalar risk score per node.

    This intentionally ignores graph structure and focuses on a
    lightweight smoke-test for survival training.
    """

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns shape (N,)
        out = self.net(x).squeeze(-1)
        return out


def cox_breslow_loss(scores: torch.Tensor, times: torch.Tensor,
                     events: torch.Tensor) -> torch.Tensor:
    """Negative partial log-likelihood (Breslow) for Cox model.

    Args:
        scores: Predicted risk scores (higher -> higher hazard), shape (N,)
        times: Observed times, shape (N,)
        events: Event indicator (1=event, 0=censored), shape (N,)

    Returns:
        Scalar loss tensor.
    """
    # Convert tensors to float and ensure 1d
    scores = scores.flatten()
    times = times.flatten().float()
    events = events.flatten().float()

    # Sort by descending time so risk set is cumulative from top
    order = torch.argsort(times, descending=True)
    scores = scores[order]
    events = events[order]

    exp_scores = torch.exp(scores)
    # cumulative sum for risk sets
    cum_exp = torch.cumsum(exp_scores, dim=0)

    # Only observed events contribute
    observed = events == 1
    if observed.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    log_risk = torch.log(cum_exp[observed])
    loss = -(scores[observed] - log_risk).sum()
    loss = loss / max(1.0, float(observed.sum()))
    return loss


def concordance_index(times: np.ndarray, events: np.ndarray,
                      preds: np.ndarray) -> float:
    """Compute Harrell's concordance index (pairwise).
    
    Robust implementation that handles edge cases and small datasets.
    Returns -1.0 if no comparable pairs exist.
    """
    n = 0
    num_correct = 0
    n_ties = 0
    N = len(times)
    
    # Ensure arrays are 1D numpy arrays
    times = np.asarray(times).flatten()
    events = np.asarray(events).flatten()
    preds = np.asarray(preds).flatten()
    
    for i in range(N):
        # Only compare pairs where we have an event
        if events[i] != 1:
            continue
            
        for j in range(N):
            if i == j:
                continue
                
            # For patient i with event, compare with patient j
            # i had event before j was observed
            if times[i] < times[j]:
                n += 1
                if preds[i] > preds[j]:
                    num_correct += 1
                elif abs(preds[i] - preds[j]) < 1e-8:
                    n_ties += 1

    if n == 0:
        return -1.0  # No comparable pairs
    return (num_correct + 0.5 * n_ties) / n


def run_training(train_path: Path, val_path: Path, test_path: Path,
                 epochs: int = 20, lr: float = 1e-3,
                 hidden: int = 64, out_dir: Path | None = None) -> None:
    """Run a short training loop and save the best checkpoint."""
    train = load_data(train_path)
    val = load_data(val_path)
    test = load_data(test_path)

    # Print dataset statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Train: {train.x.shape[0]} patients, {train.x.shape[1]} features")
    print(f"  Events: {train.event.sum().item():.0f}/{len(train.event)} ({100*train.event.mean():.1f}%)")
    print(f"  Time range: [{train.time.min():.1f}, {train.time.max():.1f}]")
    print(f"Val:   {val.x.shape[0]} patients")
    print(f"  Events: {val.event.sum().item():.0f}/{len(val.event)} ({100*val.event.mean():.1f}%)")
    print(f"  Time range: [{val.time.min():.1f}, {val.time.max():.1f}]")
    print(f"Test:  {test.x.shape[0]} patients")
    print(f"  Events: {test.event.sum().item():.0f}/{len(test.event)} ({100*test.event.mean():.1f}%)")
    print(f"  Time range: [{test.time.min():.1f}, {test.time.max():.1f}]")
    print("="*60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Use node features
    in_dim = int(train.x.shape[1])
    model = MLPRisk(in_dim=in_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    times_train = train.time.clone().detach()
    events_train = train.event.clone().detach()

    times_val = val.time.clone().detach().numpy()
    events_val = val.event.clone().detach().numpy()
    times_test = test.time.clone().detach().numpy()
    events_test = test.event.clone().detach().numpy()

    best_cidx = -math.inf
    out_dir = Path(out_dir) if out_dir is not None else Path("outputs/phase8_2_training")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        x = train.x.to(device).float()
        scores = model(x)
        loss = cox_breslow_loss(scores, times_train.to(device), events_train.to(device))
        loss.backward()
        opt.step()

        # Evaluation (no grad)
        model.eval()
        with torch.no_grad():
            val_scores = model(val.x.to(device).float()).cpu().numpy()
            test_scores = model(test.x.to(device).float()).cpu().numpy()

        cidx_val = concordance_index(times_val, events_val, -val_scores)
        cidx_test = concordance_index(times_test, events_test, -test_scores)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | "
                  f"C-idx val: {cidx_val:.4f} | C-idx test: {cidx_test:.4f}")

        # Save best
        if cidx_val > best_cidx:
            best_cidx = cidx_val
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_cindex": cidx_val,
                "test_cindex": cidx_test,
            }
            torch.save(ckpt, out_dir / "best_survival_model.pt")

    # Final evaluation and save last
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "last_survival_model.pt")
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best val C-index: {best_cidx:.4f}")
    print(f"Checkpoints saved to: {out_dir}")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path,
                        default=Path("data/03_prodromal/enhanced_training_ready/train_data.pt"))
    parser.add_argument("--val", type=Path,
                        default=Path("data/03_prodromal/enhanced_training_ready/val_data.pt"))
    parser.add_argument("--test", type=Path,
                        default=Path("data/03_prodromal/enhanced_training_ready/test_data.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase8_2_training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.train, args.val, args.test, epochs=args.epochs,
                 lr=args.lr, hidden=args.hidden, out_dir=args.out)


if __name__ == "__main__":
    main()
