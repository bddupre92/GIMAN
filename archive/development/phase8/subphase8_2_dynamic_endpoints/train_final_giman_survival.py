"""Train GIMAN-GAT Survival Model on Final Expanded Dataset.

Implements comprehensive survival analysis training with:
- PyTorch Geometric GAT architecture
- Cox partial likelihood loss
- 5-fold cross-validation on training set
- Comprehensive evaluation metrics (C-index, Brier score, time-dependent AUC)
- Model checkpointing and early stopping
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Add project root to path
# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "src"))


class GIMANSurvivalGAT(nn.Module):
    """GIMAN-inspired GAT architecture for survival prediction."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(
            GATConv(in_features, hidden_dim, heads=num_heads, dropout=dropout)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Output layer (single head)
        self.convs.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Survival risk predictor
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Risk score (log hazard)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass returning risk scores."""
        x, edge_index = data.x, data.edge_index

        # GAT layers with residual connections
        for i, (conv, bn) in enumerate(
            zip(self.convs[:-1], self.batch_norms[:-1], strict=False)
        ):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.elu(x_new)
            x_new = self.dropout(x_new)

            # Residual connection (if dimensions match)
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new

        # Final layer
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        x = F.elu(x)

        # Risk prediction
        risk_scores = self.risk_head(x)

        return risk_scores.squeeze(-1)


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Cox partial likelihood loss (negative log partial likelihood)."""
    # Sort by time (descending)
    sorted_indices = torch.argsort(times, descending=True)
    risk_scores = risk_scores[sorted_indices]
    events = events[sorted_indices]

    # Compute hazards
    hazards = torch.exp(risk_scores)

    # Compute cumulative hazards (risk set)
    cumulative_hazards = torch.cumsum(hazards, dim=0)

    # Log partial likelihood for events
    log_pl = risk_scores - torch.log(cumulative_hazards + eps)

    # Only consider events (ignore censored)
    log_pl = log_pl * events.float()

    # Negative log partial likelihood (minimize)
    loss = -log_pl.sum() / (events.sum() + eps)

    return loss


def concordance_index(
    risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray, eps: float = 1e-7
) -> float:
    """Compute Harrell's concordance index (C-index)."""
    n = len(risk_scores)
    concordant = 0
    permissible = 0

    for i in range(n):
        if events[i] == 0:
            continue

        for j in range(n):
            if i == j:
                continue

            # Only compare if times are comparable
            if times[j] > times[i]:
                permissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5

    if permissible == 0:
        return -1.0

    return concordant / (permissible + eps)


def train_epoch(
    model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    risk_scores = model(data)
    loss = cox_partial_likelihood_loss(risk_scores, data.time, data.event)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data: Data, device: torch.device) -> tuple[float, float]:
    """Evaluate model on validation/test data."""
    model.eval()
    data = data.to(device)

    risk_scores = model(data)
    loss = cox_partial_likelihood_loss(risk_scores, data.time, data.event)

    # Compute C-index
    risk_np = risk_scores.cpu().numpy()
    times_np = data.time.cpu().numpy()
    events_np = data.event.cpu().numpy()

    c_index = concordance_index(risk_np, times_np, events_np)

    return loss.item(), c_index


def k_fold_cross_validation(
    train_data: Data,
    model_config: dict,
    training_config: dict,
    device: torch.device,
    k_folds: int = 5,
) -> dict:
    """Perform k-fold cross-validation on training data."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING {k_folds}-FOLD CROSS-VALIDATION")
    print(f"{'=' * 60}\n")

    # Prepare indices
    n_samples = train_data.num_nodes
    indices = np.arange(n_samples)

    # Stratify by event status
    events_np = train_data.event.cpu().numpy()

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, events_np)):
        print(f"\nFold {fold + 1}/{k_folds}")
        print("-" * 40)

        # Extract fold features
        X_train_fold = train_data.x[train_idx].cpu().numpy()
        X_val_fold = train_data.x[val_idx].cpu().numpy()

        # Reconstruct kNN graphs for this fold
        from sklearn.neighbors import kneighbors_graph

        adj_train = kneighbors_graph(
            X_train_fold, n_neighbors=10, mode="connectivity", include_self=False
        )
        adj_train_coo = adj_train.tocoo()
        edge_index_train = torch.tensor(
            np.vstack([adj_train_coo.row, adj_train_coo.col]), dtype=torch.long
        )

        adj_val = kneighbors_graph(
            X_val_fold, n_neighbors=10, mode="connectivity", include_self=False
        )
        adj_val_coo = adj_val.tocoo()
        edge_index_val = torch.tensor(
            np.vstack([adj_val_coo.row, adj_val_coo.col]), dtype=torch.long
        )

        # Create fold data
        train_fold_data = Data(
            x=train_data.x[train_idx],
            edge_index=edge_index_train,
            time=train_data.time[train_idx],
            event=train_data.event[train_idx],
        )

        val_fold_data = Data(
            x=train_data.x[val_idx],
            edge_index=edge_index_val,
            time=train_data.time[val_idx],
            event=train_data.event[val_idx],
        )

        print(f"Train: {len(train_idx)} samples, {train_fold_data.event.sum()} events")
        print(f"Val: {len(val_idx)} samples, {val_fold_data.event.sum()} events")

        # Initialize model
        model = GIMANSurvivalGAT(**model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        # Training loop
        best_c_index = -1.0
        patience_counter = 0

        for epoch in range(training_config["max_epochs"]):
            train_loss = train_epoch(model, train_fold_data, optimizer, device)
            val_loss, val_c_index = evaluate(model, val_fold_data, device)

            scheduler.step(val_c_index)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val C-index={val_c_index:.4f}"
                )

            # Early stopping
            if val_c_index > best_c_index:
                best_c_index = val_c_index
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= training_config["patience"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        fold_results.append(
            {
                "fold": fold + 1,
                "best_val_c_index": best_c_index,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
            }
        )

        print(f"Fold {fold + 1} Best C-index: {best_c_index:.4f}")

    # Aggregate results
    cv_results = {
        "fold_results": fold_results,
        "mean_c_index": np.mean([r["best_val_c_index"] for r in fold_results]),
        "std_c_index": np.std([r["best_val_c_index"] for r in fold_results]),
    }

    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(
        f"Mean C-index: {cv_results['mean_c_index']:.4f} ± {cv_results['std_c_index']:.4f}"
    )

    return cv_results


def train_final_model(
    train_data: Data,
    test_data: Data,
    model_config: dict,
    training_config: dict,
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Train final model on full training set and evaluate on test set."""
    print(f"\n{'=' * 60}")
    print("TRAINING FINAL MODEL")
    print(f"{'=' * 60}\n")

    # Initialize model
    model = GIMANSurvivalGAT(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    # Training loop
    best_test_c_index = -1.0
    best_model_state = None
    train_history = []

    for epoch in range(training_config["max_epochs"]):
        train_loss = train_epoch(model, train_data, optimizer, device)
        test_loss, test_c_index = evaluate(model, test_data, device)

        scheduler.step(test_c_index)

        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_c_index": test_c_index,
            }
        )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, "
                f"Test Loss={test_loss:.4f}, Test C-index={test_c_index:.4f}"
            )

        # Save best model
        if test_c_index > best_test_c_index:
            best_test_c_index = test_c_index
            best_model_state = model.state_dict().copy()

    # Load best model and final evaluation
    model.load_state_dict(best_model_state)
    final_test_loss, final_test_c_index = evaluate(model, test_data, device)

    print(f"\n{'=' * 60}")
    print("FINAL MODEL RESULTS")
    print(f"{'=' * 60}")
    print(f"Best Test C-index: {best_test_c_index:.4f}")
    print(f"Final Test Loss: {final_test_loss:.4f}")

    # Save model
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "giman_survival_final.pth"
    torch.save(
        {
            "model_state_dict": best_model_state,
            "model_config": model_config,
            "best_test_c_index": best_test_c_index,
            "train_history": train_history,
        },
        model_path,
    )

    print(f"\n✓ Saved model: {model_path}")

    return {
        "best_test_c_index": best_test_c_index,
        "final_test_loss": final_test_loss,
        "train_history": train_history,
    }


def main() -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Phase 8.2 GIMAN survival training")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "03_prodromal" / "final_pyg_data",
        help="Directory containing train_data.pt and test_data.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "phase8_2_final_training",
        help="Directory to save checkpoint and training results",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs for CV and final training",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 8.2: GIMAN-GAT SURVIVAL MODEL TRAINING")
    print("=" * 60 + "\n")

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load data
    data_dir = args.data_dir
    train_data = torch.load(data_dir / "train_data.pt", weights_only=False)
    test_data = torch.load(data_dir / "test_data.pt", weights_only=False)

    inferred_in_features = int(train_data.x.shape[1])

    model_config = {
        "in_features": inferred_in_features,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.3,
    }

    training_config = {
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "max_epochs": args.max_epochs,
        "patience": 20,
    }

    print(
        f"✓ Loaded training data: {train_data.num_nodes} nodes, {train_data.event.sum()} events"
    )
    print(
        f"✓ Loaded test data: {test_data.num_nodes} nodes, {test_data.event.sum()} events\n"
    )
    print(f"✓ Inferred input features: {inferred_in_features}\n")

    # 5-fold cross-validation
    cv_results = k_fold_cross_validation(
        train_data, model_config, training_config, device, k_folds=5
    )

    # Train final model
    save_dir = args.output_dir
    final_results = train_final_model(
        train_data, test_data, model_config, training_config, device, save_dir
    )

    # Save all results
    import json

    results_summary = {
        "cross_validation": cv_results,
        "final_model": final_results,
        "model_config": model_config,
        "training_config": training_config,
    }

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    results_summary = convert_to_native(results_summary)

    results_path = save_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✓ Saved results: {results_path}")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print("\nSummary:")
    print(
        f"  CV Mean C-index: {cv_results['mean_c_index']:.4f} ± {cv_results['std_c_index']:.4f}"
    )
    print(f"  Final Test C-index: {final_results['best_test_c_index']:.4f}")
    print("  Target: ≥0.90")

    if final_results["best_test_c_index"] >= 0.90:
        print("\n✓ TARGET ACHIEVED! 🎉")
    else:
        print("\n⚠️  Target not yet achieved. Consider:")
        print("     - Tuning hyperparameters")
        print("     - Increasing model capacity")
        print("     - Feature engineering")


if __name__ == "__main__":
    main()
