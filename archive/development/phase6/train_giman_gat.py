"""
Train GIMAN-GAT Model on Real PPMI Data

This script trains the GAT-based GIMAN model on the enhanced PPMI dataset,
maintaining compatibility with existing training infrastructure while enabling
native attention mechanisms for Phase 6 explainability.

Author: AI Research Assistant
Date: October 5, 2025
Context: Phase 6 GNN Explainability - GAT Training
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from archive.development.phase6.task_6_0_1_gat_upgrade import GIMANBackboneGAT

warnings.filterwarnings("ignore")


class GIMANGATTrainer:
    """
    Train GIMAN-GAT model with comprehensive evaluation and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        output_dir: str = "models/giman_gat",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GAT trainer.

        Args:
            model: GIMANBackboneGAT model
            data: PyG Data object with graph structure
            output_dir: Directory for model checkpoints and results
            device: Training device (cuda/cpu)
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_auc': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.patience_counter = 0

        print(f"[INIT] GIMAN-GAT Trainer")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Training samples: {data.num_nodes}")
        print(f"   Output directory: {output_dir}")

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_mask: torch.Tensor
    ) -> tuple:
        """
        Train for one epoch.

        Args:
            optimizer: Optimizer
            criterion: Loss function
            train_mask: Boolean mask for training nodes

        Returns:
            (loss, accuracy) for training set
        """
        self.model.train()
        optimizer.zero_grad()

        # Forward pass
        output = self.model(
            self.data.x,
            self.data.edge_index,
            batch=self.data.batch if hasattr(self.data, 'batch') else None
        )

        logits = output['logits']

        # Compute loss on training nodes
        loss = criterion(logits[train_mask], self.data.y[train_mask])

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            pred = logits[train_mask].argmax(dim=1)
            acc = accuracy_score(
                self.data.y[train_mask].cpu().numpy(),
                pred.cpu().numpy()
            )

        return loss.item(), acc

    @torch.no_grad()
    def evaluate(
        self,
        criterion: nn.Module,
        val_mask: torch.Tensor
    ) -> tuple:
        """
        Evaluate model on validation set.

        Args:
            criterion: Loss function
            val_mask: Boolean mask for validation nodes

        Returns:
            (loss, accuracy, auc) for validation set
        """
        self.model.eval()

        # Forward pass
        output = self.model(
            self.data.x,
            self.data.edge_index,
            batch=self.data.batch if hasattr(self.data, 'batch') else None
        )

        logits = output['logits']

        # Compute loss
        loss = criterion(logits[val_mask], self.data.y[val_mask])

        # Compute metrics
        pred = logits[val_mask].argmax(dim=1)
        probs = F.softmax(logits[val_mask], dim=1)[:, 1]

        acc = accuracy_score(
            self.data.y[val_mask].cpu().numpy(),
            pred.cpu().numpy()
        )

        try:
            auc = roc_auc_score(
                self.data.y[val_mask].cpu().numpy(),
                probs.cpu().numpy()
            )
        except:
            auc = 0.5  # Fallback if AUC can't be computed

        return loss.item(), acc, auc

    def train(
        self,
        epochs: int = 200,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        patience: int = 30,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """
        Full training loop with early stopping.

        Args:
            epochs: Maximum number of epochs
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
        """
        print("\n" + "="*80)
        print("TRAINING GIMAN-GAT MODEL")
        print("="*80 + "\n")

        # Create train/val/test split
        num_nodes = self.data.num_nodes
        indices = torch.randperm(num_nodes)

        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        print(f"Data Split:")
        print(f"   Training: {train_mask.sum()} nodes ({train_ratio*100:.0f}%)")
        print(f"   Validation: {val_mask.sum()} nodes ({val_ratio*100:.0f}%)")
        print(f"   Test: {test_mask.sum()} nodes ({(1-train_ratio-val_ratio)*100:.0f}%)")

        # Initialize optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        criterion = nn.CrossEntropyLoss()

        print(f"\nTraining Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Early stopping patience: {patience}")
        print(f"   Device: {self.device}")

        # Training loop
        print("\n" + "-"*80)
        print("Starting training...")
        print("-"*80)

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion, train_mask)

            # Validate
            val_loss, val_acc, val_auc = self.evaluate(criterion, val_mask)

            # Update learning rate
            scheduler.step(val_loss)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_auc)
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation AUC: {self.best_val_auc:.4f}")
                break

        # Final evaluation on test set
        print("\n" + "-"*80)
        print("Training complete! Evaluating on test set...")
        print("-"*80)

        test_loss, test_acc, test_auc = self.evaluate(criterion, test_mask)

        print(f"\nFinal Test Results:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   AUC-ROC: {test_auc:.4f}")

        # Save final model and history
        self.save_checkpoint('final_model.pth', epochs, test_auc)
        self.save_training_history()

        print(f"\n[COMPLETE] Models saved to {self.output_dir}")

    def save_checkpoint(self, filename: str, epoch: int, auc: float):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'output_dim': self.model.output_dim,
                'num_heads': self.model.num_heads,
                'dropout_rate': self.model.dropout_rate,
                'pooling_method': self.model.pooling_method
            },
            'auc': auc,
            'history': self.history
        }, checkpoint_path)

        print(f"   Checkpoint saved: {filename} (AUC: {auc:.4f})")

    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"   Training history saved: {history_path}")


def load_ppmi_data(data_path: str = "data/enhanced/enhanced_graph_data_latest.pth"):
    """Load enhanced PPMI graph data."""
    print(f"\n[LOAD] Loading PPMI data from {data_path}...")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = torch.load(data_path)

    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Features: {data.num_node_features}")
    print(f"   Classes: {data.y.unique().tolist() if hasattr(data, 'y') else 'N/A'}")

    return data


def main():
    """Main training execution."""
    print("\n" + "="*80)
    print("GIMAN-GAT TRAINING PIPELINE")
    print("="*80)

    # Configuration
    DATA_PATH = "data/enhanced/enhanced_graph_data_fixed_20250924_084000.pth"
    OUTPUT_DIR = "models/giman_gat"

    EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    PATIENCE = 30

    # Load data
    data = load_ppmi_data(DATA_PATH)

    # Initialize model
    print("\n[INIT] Initializing GIMAN-GAT model...")
    model = GIMANBackboneGAT(
        input_dim=data.num_node_features,
        hidden_dims=[64, 128, 64],
        output_dim=2,  # Binary classification
        num_heads=4,
        dropout_rate=0.3,
        attention_dropout=0.1,
        pooling_method='concat',
        use_residual=True,
        concat_heads=True,
        classification_level='node'  # Node-level classification
    )

    # Initialize trainer
    trainer = GIMANGATTrainer(
        model=model,
        data=data,
        output_dir=OUTPUT_DIR
    )

    # Train model
    trainer.train(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest Model:")
    print(f"   Path: {OUTPUT_DIR}/best_model.pth")
    print(f"   Validation AUC: {trainer.best_val_auc:.4f}")
    print(f"\nNext Steps:")
    print(f"   1. Review training history: {OUTPUT_DIR}/training_history.json")
    print(f"   2. Proceed to Task 6.1: Attention Weight Visualization")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
