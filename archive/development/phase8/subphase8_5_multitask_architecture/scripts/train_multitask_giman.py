"""
PHASE 8.5: Multi-Task GIMAN Training Pipeline

Train a unified GIMAN model with shared encoder and 4 task-specific heads for:
    - Task 1: Progression prediction (survival)
    - Task 2: Conversion prediction (survival)
    - Task 3: SAA prediction (binary classification)
    - Task 4: Diagnostic prediction (2-class classification)

Author: GIMAN Research Team
Date: October 15, 2025
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np

# Add models directory to path
models_dir = Path(__file__).resolve().parent.parent / "models"
sys.path.insert(0, str(models_dir))

from giman_multitask import GIMANMultiTask, create_multitask_giman
from multitask_loss import MultiTaskLoss


def concordance_index(
    log_hazards: torch.Tensor,
    survival_time: torch.Tensor,
    event_observed: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute Harrell's C-index for survival analysis.
    
    Args:
        log_hazards: Predicted log-hazard scores
        survival_time: Time to event or censoring
        event_observed: Binary indicator (1=event, 0=censored)
        mask: Boolean mask for valid labels
        
    Returns:
        c_index: Concordance index [0, 1]
    """
    if mask is not None:
        log_hazards = log_hazards[mask].squeeze(-1).detach().cpu().numpy()
        survival_time = survival_time[mask].detach().cpu().numpy()
        event_observed = event_observed[mask].detach().cpu().numpy()
    else:
        log_hazards = log_hazards.squeeze(-1).detach().cpu().numpy()
        survival_time = survival_time.detach().cpu().numpy()
        event_observed = event_observed.detach().cpu().numpy()
    
    if len(log_hazards) < 2 or event_observed.sum() == 0:
        return 0.5  # Default for insufficient data
    
    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    ties_risk = 0
    
    for i in range(len(log_hazards)):
        if event_observed[i] == 0:  # Skip censored as reference
            continue
        
        for j in range(len(log_hazards)):
            if i == j:
                continue
            
            # Only consider pairs where i is observed event
            if survival_time[i] < survival_time[j]:
                # i occurred before j
                if log_hazards[i] > log_hazards[j]:
                    concordant += 1
                elif log_hazards[i] < log_hazards[j]:
                    discordant += 1
                else:
                    ties_risk += 1
    
    total_pairs = concordant + discordant + ties_risk
    if total_pairs == 0:
        return 0.5
    
    c_index = (concordant + 0.5 * ties_risk) / total_pairs
    return c_index


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    data: Data
) -> Dict[str, float]:
    """
    Compute task-specific metrics.
    
    Args:
        predictions: Model predictions
        data: PyG Data object with labels and masks
        
    Returns:
        metrics: Dictionary of task metrics
    """
    metrics = {}
    
    # Task 1: Progression C-index
    if 'progression' in predictions:
        prog_mask = data.progression_mask.bool()
        if prog_mask.sum() > 0:
            c_index = concordance_index(
                predictions['progression'],
                data.progression_time,
                data.progression_event,
                prog_mask
            )
            metrics['progression_c_index'] = c_index
    
    # Task 2: Conversion metrics (Binary Classification)
    if 'conversion' in predictions:
        conv_mask = data.conversion_mask.bool()
        if conv_mask.sum() > 0:
            logits = predictions['conversion'][conv_mask].squeeze(-1).detach().cpu()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)
            targets = data.conversion_label[conv_mask].detach().cpu().numpy()
            
            try:
                auc = roc_auc_score(targets, probs)
                metrics['conversion_auc'] = auc
            except:
                metrics['conversion_auc'] = 0.5
            
            metrics['conversion_accuracy'] = accuracy_score(targets, preds)
            metrics['conversion_f1'] = f1_score(targets, preds, zero_division=0)
    
    # Task 3: SAA metrics (AUC, Accuracy, F1)
    if 'saa' in predictions:
        saa_mask = data.saa_mask.bool()
        if saa_mask.sum() > 0:
            logits = predictions['saa'][saa_mask].squeeze(-1).detach().cpu()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)
            targets = data.saa_label[saa_mask].detach().cpu().numpy()
            
            try:
                auc = roc_auc_score(targets, probs)
                metrics['saa_auc'] = auc
            except:
                metrics['saa_auc'] = 0.5
            
            metrics['saa_accuracy'] = accuracy_score(targets, preds)
            metrics['saa_f1'] = f1_score(targets, preds, zero_division=0)
    
    # Task 4: Diagnostic metrics (Accuracy, F1)
    if 'diagnostic' in predictions:
        diag_mask = data.diagnostic_mask.bool()
        if diag_mask.sum() > 0:
            logits = predictions['diagnostic'][diag_mask].detach().cpu()
            preds = torch.argmax(logits, dim=1).numpy()
            targets = data.diagnostic_label[diag_mask].detach().cpu().numpy()
            
            metrics['diagnostic_accuracy'] = accuracy_score(targets, preds)
            metrics['diagnostic_f1'] = f1_score(targets, preds, average='weighted', zero_division=0)
    
    return metrics


class MultiTaskTrainer:
    """
    Trainer for multi-task GIMAN model.
    
    Args:
        model: GIMANMultiTask model
        loss_fn: MultiTaskLoss function
        optimizer: PyTorch optimizer
        device: Training device
        output_dir: Directory for checkpoints and logs
    """
    
    def __init__(
        self,
        model: GIMANMultiTask,
        loss_fn: MultiTaskLoss,
        optimizer: optim.Optimizer,
        device: torch.device,
        output_dir: Path,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.scheduler = scheduler
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, data: Data) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass
        predictions = self.model(data)
        
        # Compute loss
        total_loss, task_losses = self.loss_fn(predictions, data)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(predictions, data)
        
        return total_loss.item(), metrics
    
    def validate(self, data: Data) -> Tuple[float, Dict[str, float]]:
        """Validate on validation set."""
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            predictions = self.model(data)
            
            # Compute loss
            total_loss, task_losses = self.loss_fn(predictions, data)
            
            # Compute metrics
            metrics = compute_metrics(predictions, data)
        
        return total_loss.item(), metrics
    
    def train(
        self,
        train_data: Data,
        val_data: Data,
        num_epochs: int = 100,
        patience: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Train multi-task model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Print training progress
            
        Returns:
            history: Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_data)
            self.history['train_loss'].append(train_loss)
            self.history['train_metrics'].append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate(val_data)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Print task-specific metrics
                for task in ['progression', 'conversion', 'saa', 'diagnostic']:
                    train_str = ""
                    val_str = ""
                    
                    if task == 'progression':
                        metric_name = f"{task}_c_index"
                        if metric_name in train_metrics:
                            train_str = f"C-index={train_metrics[metric_name]:.4f}"
                        if metric_name in val_metrics:
                            val_str = f"C-index={val_metrics[metric_name]:.4f}"
                        if train_str or val_str:
                            print(f"  {task.capitalize()}: {train_str} | {val_str}")
                    
                    elif task == 'conversion':
                        if 'conversion_auc' in train_metrics:
                            train_str = f"AUC={train_metrics['conversion_auc']:.4f}"
                        if 'conversion_auc' in val_metrics:
                            val_str = f"AUC={val_metrics['conversion_auc']:.4f}"
                        if train_str or val_str:
                            print(f"  {task.capitalize()}: {train_str} | {val_str}")
                    
                    elif task == 'saa':
                        if 'saa_auc' in train_metrics:
                            train_str = f"AUC={train_metrics['saa_auc']:.4f}"
                        if 'saa_auc' in val_metrics:
                            val_str = f"AUC={val_metrics['saa_auc']:.4f}"
                        if train_str or val_str:
                            print(f"  SAA: {train_str} | {val_str}")
                    
                    elif task == 'diagnostic':
                        if 'diagnostic_accuracy' in train_metrics:
                            train_str = f"Acc={train_metrics['diagnostic_accuracy']:.4f}"
                        if 'diagnostic_accuracy' in val_metrics:
                            val_str = f"Acc={val_metrics['diagnostic_accuracy']:.4f}"
                        if train_str or val_str:
                            print(f"  Diagnostic: {train_str} | {val_str}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'history': self.history
                }
                torch.save(checkpoint, self.output_dir / 'best_multitask_model.pth')
                
                if verbose:
                    print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Save final training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def main():
    """Main training function."""
    print("=" * 80)
    print("PHASE 8.5: MULTI-TASK GIMAN TRAINING")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Paths
    project_root = Path(__file__).resolve().parents[5]  # Go up to project root
    data_dir = project_root / "archive" / "development" / "phase8" / "subphase8_5_multitask_architecture" / "data"
    output_dir = project_root / "archive" / "development" / "phase8" / "subphase8_5_multitask_architecture" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_data_list = torch.load(data_dir / "multitask_train_data.pt", weights_only=False)
    val_data_list = torch.load(data_dir / "multitask_val_data.pt", weights_only=False)
    test_data_list = torch.load(data_dir / "multitask_test_data.pt", weights_only=False)
    
    # Unwrap from list format
    train_data = train_data_list[0] if isinstance(train_data_list, list) else train_data_list
    val_data = val_data_list[0] if isinstance(val_data_list, list) else val_data_list
    test_data = test_data_list[0] if isinstance(test_data_list, list) else test_data_list
    
    print(f"  Train: {train_data.num_nodes} nodes")
    print(f"  Val:   {val_data.num_nodes} nodes")
    print(f"  Test:  {test_data.num_nodes} nodes")
    
    # Create model
    print("\n[2/5] Creating model...")
    model = create_multitask_giman(input_dim=49, hidden_dim=128)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")
    
    # Create loss function with adjusted task weights
    print("\n[3/5] Setting up loss and optimizer...")
    print("  IMPROVED CONFIG:")
    print("    - Task weights: Progression=10.0, Conversion=10.0, SAA=1.5, Diagnostic=1.0")
    print("    - Conversion pos_weight: 1.5 → 4.0 (to handle 6/24 class imbalance)")
    print("    - Training epochs: 100 → 200 with patience=40")
    print("    - Learning rate: 0.001 with ReduceLROnPlateau scheduler")
    
    diagnostic_weights = torch.tensor([0.132, 0.868])  # From data prep
    loss_fn = MultiTaskLoss(
        task_weights={
            'progression': 10.0,   # ↑ from 1.0 - Compensate for limited labels
            'conversion': 10.0,    # ↑ from 1.0 - Compensate for limited labels
            'saa': 1.5,            # ↑ from 1.0 - Slight boost
            'diagnostic': 1.0      # Keep baseline
        },
        saa_pos_weight=4.63,
        conversion_pos_weight=4.0,  # ↑ from 1.5 - Handle 24/6 imbalance
        diagnostic_class_weights=diagnostic_weights
    )
    
    # Create optimizer with learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    
    # Create trainer
    print("\n[4/5] Training model...")
    trainer = MultiTaskTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        scheduler=scheduler  # Add scheduler
    )
    
    # Train with extended epochs and patience
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=200,      # ↑ from 100
        patience=40,         # ↑ from 20
        verbose=True
    )
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    model.load_state_dict(
        torch.load(output_dir / 'best_multitask_model.pth')['model_state_dict']
    )
    test_loss, test_metrics = trainer.validate(test_data)
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_metrics': test_metrics
    }
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - best_multitask_model.pth")
    print(f"  - training_history.json")
    print(f"  - test_results.json")


if __name__ == "__main__":
    main()
