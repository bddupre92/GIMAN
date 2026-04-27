"""
Phase 6 Task 6.0.3: Train GAT Models for Prognostic Tasks

Trains separate GAT models for:
1. Phase 4: Subtype classification (3-class: slow/moderate/fast progressors)
2. Phase 5: Prodromal conversion prediction (2-class: converter/non-converter)

Author: GIMAN Development Team
Date: October 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from archive.development.phase6.task_6_0_1_gat_upgrade import GIMANBackboneGAT


class PrognosticGATTrainer:
    """Train GAT model for prognostic tasks (Phase 4 or Phase 5)"""

    def __init__(
        self,
        data: Data,
        metadata: dict,
        model_save_dir: str,
        task_name: str = "prognostic"
    ):
        self.data = data
        self.metadata = metadata
        self.task_name = task_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create save directory
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Move data to device
        self.data = self.data.to(self.device)

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

        # Best model tracking
        self.best_val_auc = 0.0
        self.best_val_acc = 0.0
        self.patience_counter = 0

        print(f"\n{'='*60}")
        print(f"PROGNOSTIC GAT TRAINER: {task_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Task: {metadata['task']}")
        print(f"Patients: {metadata['num_patients']}")
        print(f"Features: {metadata['num_features']}")
        print(f"Classes: {data.num_classes}")
        print(f"Edges: {metadata['num_edges']}")

    def create_model(self, hidden_dims=[64, 128, 64], num_heads=4, dropout=0.3):
        """Create GAT model"""
        model = GIMANBackboneGAT(
            input_dim=self.data.num_node_features,
            hidden_dims=hidden_dims,
            output_dim=self.data.num_classes,
            num_heads=num_heads,
            dropout_rate=dropout,
            attention_dropout=0.1,
            pooling_method='concat',
            use_residual=True,
            concat_heads=True,
            classification_level='node'  # Patient-level classification
        ).to(self.device)

        print(f"\nModel created:")
        print(f"  Input: {self.data.num_node_features} features")
        print(f"  Hidden: {hidden_dims}")
        print(f"  Output: {self.data.num_classes} classes")
        print(f"  Attention heads: {num_heads}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def train_epoch(self, model, optimizer, criterion, train_mask):
        """Single training epoch"""
        model.train()
        optimizer.zero_grad()

        # Forward pass
        model_out = model(self.data.x, self.data.edge_index)
        out = model_out['logits'] if isinstance(model_out, dict) else model_out

        # Compute loss on training nodes
        loss = criterion(out[train_mask], self.data.y[train_mask])

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute accuracy
        pred = out[train_mask].argmax(dim=1)
        acc = accuracy_score(
            self.data.y[train_mask].cpu().numpy(),
            pred.cpu().numpy()
        )

        return loss.item(), acc

    def evaluate(self, model, criterion, mask):
        """Evaluate model on given mask"""
        model.eval()

        with torch.no_grad():
            model_out = model(self.data.x, self.data.edge_index)
            out = model_out['logits'] if isinstance(model_out, dict) else model_out

            # Compute loss on masked nodes
            loss = criterion(out[mask], self.data.y[mask])

            # Predictions
            pred = out[mask].argmax(dim=1)
            proba = F.softmax(out[mask], dim=1)

            # Metrics
            y_true = self.data.y[mask].cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_proba = proba.cpu().numpy()

            acc = accuracy_score(y_true, y_pred)

            # AUC (handle binary and multiclass)
            if self.data.num_classes == 2:
                try:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                except:
                    auc = 0.5  # Fallback for edge cases
            else:
                try:
                    auc = roc_auc_score(
                        y_true, y_proba,
                        multi_class='ovr', average='macro'
                    )
                except:
                    auc = 0.0  # Fallback if AUC computation fails

        return loss.item(), acc, auc

    def train(
        self,
        epochs=200,
        lr=0.001,
        weight_decay=5e-4,
        patience=30,
        train_ratio=0.7,
        val_ratio=0.15
    ):
        """Train GAT model with early stopping"""

        print(f"\n{'='*60}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {weight_decay}")
        print(f"Patience: {patience}")
        print(f"Train/Val/Test split: {train_ratio}/{val_ratio}/{1-train_ratio-val_ratio}")

        # Create model
        model = self.create_model()

        # Create train/val/test split (stratified)
        num_nodes = self.data.num_nodes
        indices = np.arange(num_nodes)

        # Stratified split
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=(1 - train_ratio - val_ratio),
            stratify=self.data.y.cpu().numpy(),
            random_state=42
        )

        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio / (train_ratio + val_ratio),
            stratify=self.data.y[train_val_idx].cpu().numpy(),
            random_state=42
        )

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        print(f"\nData split:")
        print(f"  Train: {train_mask.sum().item()} nodes ({train_mask.sum().item()/num_nodes*100:.1f}%)")
        print(f"  Val:   {val_mask.sum().item()} nodes ({val_mask.sum().item()/num_nodes*100:.1f}%)")
        print(f"  Test:  {test_mask.sum().item()} nodes ({test_mask.sum().item()/num_nodes*100:.1f}%)")

        # Class distribution in splits
        print(f"\nClass distribution:")
        for split_name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
            class_counts = torch.bincount(self.data.y[mask])
            print(f"  {split_name}: {class_counts.cpu().numpy()}")

        # Handle class imbalance with weighted loss
        class_counts = torch.bincount(self.data.y[train_mask])
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(self.device)

        print(f"\nClass weights: {class_weights.cpu().numpy()}")

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )

        # Training loop
        print(f"\n{'='*60}")
        print("TRAINING")
        print(f"{'='*60}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(model, optimizer, criterion, train_mask)
            val_loss, val_acc, val_auc = self.evaluate(model, criterion, val_mask)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_acc = val_acc
                self.save_checkpoint(model, 'best_model.pth', epoch, val_auc, val_acc)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best Val AUC: {self.best_val_auc:.4f}")
                break

            # Learning rate scheduling
            scheduler.step(val_auc)

        # Load best model and evaluate on test set
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")

        checkpoint = torch.load(self.model_save_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_auc = self.evaluate(model, criterion, test_mask)

        print(f"\nBest Model Performance:")
        print(f"  Validation AUC: {self.best_val_auc:.4f}")
        print(f"  Validation Acc: {self.best_val_acc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")

        # Detailed test set analysis
        model.eval()
        with torch.no_grad():
            model_out = model(self.data.x, self.data.edge_index)
            out = model_out['logits'] if isinstance(model_out, dict) else model_out
            test_pred = out[test_mask].argmax(dim=1).cpu().numpy()
            test_true = self.data.y[test_mask].cpu().numpy()

        print(f"\nClassification Report (Test Set):")
        print(classification_report(test_true, test_pred, zero_division=0))

        print(f"\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(test_true, test_pred)
        print(cm)

        # Save results (convert numpy arrays to lists for JSON serialization)
        metadata_json = {}
        for key, value in self.metadata.items():
            if isinstance(value, np.ndarray):
                metadata_json[key] = value.tolist()
            elif isinstance(value, dict):
                metadata_json[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in value.items()}
            else:
                metadata_json[key] = value

        results = {
            'task': self.metadata['task'],
            'best_val_auc': float(self.best_val_auc),
            'best_val_acc': float(self.best_val_acc),
            'test_auc': float(test_auc),
            'test_acc': float(test_acc),
            'num_epochs': len(self.history['train_loss']),
            'metadata': metadata_json
        }

        with open(self.model_save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Plot training curves
        self.plot_training_curves()

        return model, results

    def save_checkpoint(self, model, filename, epoch, val_auc, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_auc': val_auc,
            'val_acc': val_acc,
            'metadata': self.metadata,
            'history': self.history
        }
        torch.save(checkpoint, self.model_save_dir / filename)

    def plot_training_curves(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # AUC
        axes[2].plot(self.history['val_auc'], label='Val AUC', color='green')
        axes[2].axhline(y=self.best_val_auc, color='red', linestyle='--', label=f'Best: {self.best_val_auc:.4f}')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('Validation AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nTraining curves saved to: {self.model_save_dir / 'training_curves.png'}")


def main():
    """Train both Phase 4 and Phase 5 GAT models"""

    base_path = Path("e:/My Drive/CSCI FALL 2025")
    data_dir = base_path / "data/prognostic_graphs"
    models_dir = base_path / "models"

    print("\n" + "="*60)
    print("PROGNOSTIC GAT MODEL TRAINING")
    print("="*60)

    # Train Phase 4 model (Subtype classification)
    print("\n\n" + "#"*60)
    print("# PHASE 4: SUBTYPE CLASSIFICATION (3-class)")
    print("#"*60)

    try:
        # Load Phase 4 data
        phase4_path = data_dir / "phase4_subtype_graph.pth"
        phase4_loaded = torch.load(phase4_path)
        phase4_data = phase4_loaded['data']
        phase4_meta = phase4_loaded['metadata']

        # Train Phase 4 model
        phase4_trainer = PrognosticGATTrainer(
            data=phase4_data,
            metadata=phase4_meta,
            model_save_dir=str(models_dir / "giman_gat_phase4"),
            task_name="Phase 4 Subtype Classification"
        )

        phase4_model, phase4_results = phase4_trainer.train(
            epochs=200,
            lr=0.001,
            weight_decay=5e-4,
            patience=30
        )

        print(f"\nPhase 4 training complete!")
        print(f"Model saved to: {models_dir / 'giman_gat_phase4'}")

    except Exception as e:
        print(f"\nERROR training Phase 4 model: {e}")
        import traceback
        traceback.print_exc()

    # Train Phase 5 model (Prodromal conversion)
    print("\n\n" + "#"*60)
    print("# PHASE 5: PRODROMAL CONVERSION PREDICTION (2-class)")
    print("#"*60)

    try:
        # Load Phase 5 data
        phase5_path = data_dir / "phase5_conversion_graph.pth"
        phase5_loaded = torch.load(phase5_path)
        phase5_data = phase5_loaded['data']
        phase5_meta = phase5_loaded['metadata']

        # Train Phase 5 model
        phase5_trainer = PrognosticGATTrainer(
            data=phase5_data,
            metadata=phase5_meta,
            model_save_dir=str(models_dir / "giman_gat_phase5"),
            task_name="Phase 5 Prodromal Conversion"
        )

        phase5_model, phase5_results = phase5_trainer.train(
            epochs=200,
            lr=0.001,
            weight_decay=5e-4,
            patience=30
        )

        print(f"\nPhase 5 training complete!")
        print(f"Model saved to: {models_dir / 'giman_gat_phase5'}")

    except Exception as e:
        print(f"\nERROR training Phase 5 model: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("PROGNOSTIC GAT TRAINING COMPLETE")
    print("="*60)
    print("\nModels saved:")
    print(f"  Phase 4 (Subtype): {models_dir / 'giman_gat_phase4'}")
    print(f"  Phase 5 (Conversion): {models_dir / 'giman_gat_phase5'}")
    print("\nNext step: Apply Task 6.1 attention visualization to all GAT models")


if __name__ == "__main__":
    main()
