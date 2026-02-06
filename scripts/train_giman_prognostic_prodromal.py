"""
Phase 8.1 Task 12: Train GIMAN-Prognostic on Prodromal Cohort

Trains GIMAN on prodromal cohort (n=381) with REAL phenoconversion events.
Uses CoxPH loss for time-to-event prediction.

Key comparison: Manifest PD test C-index 0.38 vs Prodromal target ≥0.55

Author: GIMAN Research Team  
Date: October 12, 2025
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from models.giman_progression import GIMANProgression

# ============================================================================
# CONFIGURATION  
# ============================================================================

DATA_DIR = Path("data/03_prodromal/training_ready")
OUTPUT_DIR = Path("results/phase8_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model architecture (identical to manifest PD for fair comparison)
NUM_FEATURES = 4  # AGE, SEX, UPDRS, MoCA (only available features)
HIDDEN_DIM = 64
NUM_GAT_LAYERS = 3
NUM_HEADS = 4
DROPOUT = 0.3

# Training parameters
MAX_EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n{'=' * 70}")
print("PHASE 8.1 TASK 12: TRAIN GIMAN-PROGNOSTIC ON PRODROMAL COHORT")
print(f"{'=' * 70}")
print(f"Device: {DEVICE}")
print(f"Architecture: {NUM_FEATURES} features → {NUM_GAT_LAYERS} GAT layers ({NUM_HEADS} heads) → {HIDDEN_DIM} hidden")
print(f"Training: Max {MAX_EPOCHS} epochs, patience {PATIENCE}, lr {LEARNING_RATE}")
print(f"Output: {OUTPUT_DIR}")


# ============================================================================
# LOAD DATA
# ============================================================================

def load_training_data():
    """Load train/val/test PyG Data objects."""
    print(f"\n{'=' * 70}")
    print("LOADING PRODROMAL TRAINING DATA")
    print(f"{'=' * 70}")
    
    train_data = torch.load(DATA_DIR / "train_data.pt", map_location='cpu', weights_only=False)
    val_data = torch.load(DATA_DIR / "val_data.pt", map_location='cpu', weights_only=False)
    test_data = torch.load(DATA_DIR / "test_data.pt", map_location='cpu', weights_only=False)
    
    # Load split info
    with open(DATA_DIR / "split_info.json", 'r') as f:
        split_info = json.load(f)
    
    print(f"\n✓ Loaded training data:")
    print(f"  Train: {train_data.x.shape[0]} patients, {train_data.x.shape[1]} features, {split_info['train_events']} events")
    print(f"  Val: {val_data.x.shape[0]} patients, {val_data.x.shape[1]} features, {split_info['val_events']} events")
    print(f"  Test: {test_data.x.shape[0]} patients, {test_data.x.shape[1]} features, {split_info['test_events']} events")
    
    return train_data, val_data, test_data, split_info


# ============================================================================
# COX PROPORTIONAL HAZARDS LOSS
# ============================================================================

def cox_ph_loss(risk_pred, time, event):
    """
    Cox proportional hazards partial likelihood loss.
    
    Args:
        risk_pred: Predicted risk scores (n,)
        time: Time to event/censoring (n,)
        event: Event indicator (n,)
    
    Returns:
        Negative partial log-likelihood
    """
    # Sort by time (descending)
    sorted_indices = torch.argsort(time, descending=True)
    risk_pred = risk_pred[sorted_indices]
    event = event[sorted_indices]
    
    # Compute hazard ratios
    hazard_ratio = torch.exp(risk_pred)
    
    # Compute cumulative sum (risk set)
    log_risk = torch.logcumsumexp(risk_pred, dim=0)
    
    # Compute partial likelihood for events only
    uncensored_likelihood = risk_pred - log_risk
    censored_likelihood = uncensored_likelihood * event
    
    # Negative log-likelihood
    num_events = event.sum()
    if num_events > 0:
        loss = -censored_likelihood.sum() / num_events
    else:
        loss = torch.tensor(0.0, device=risk_pred.device)
    
    return loss


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, data, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    # Move data to device
    data = data.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    risk_pred = model(data.x, data.edge_index)
    
    # Compute loss
    loss = cox_ph_loss(risk_pred.squeeze(), data.time, data.event)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device):
    """Evaluate model with C-index."""
    model.eval()
    
    # Move data to device
    data = data.to(device)
    
    # Forward pass
    risk_pred = model(data.x, data.edge_index)
    
    # Compute loss
    loss = cox_ph_loss(risk_pred.squeeze(), data.time, data.event)
    
    # Compute C-index
    risk_scores = risk_pred.squeeze().cpu().numpy()
    times = data.time.cpu().numpy()
    events = data.event.cpu().numpy()
    
    # C-index requires at least one event
    if events.sum() > 0:
        c_index = concordance_index(times, -risk_scores, events)
    else:
        c_index = 0.5
    
    return loss.item(), c_index


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function."""
    
    # Load data
    train_data, val_data, test_data, split_info = load_training_data()
    
    # Initialize model
    print(f"\n{'=' * 70}")
    print("INITIALIZING MODEL")
    print(f"{'=' * 70}")
    
    model = GIMANProgression(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_gat_layers=NUM_GAT_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model initialized:")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Architecture: {model}")
    
    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_c_index': [],
        'learning_rate': []
    }
    
    best_val_c_index = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}\n")
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_data, optimizer, DEVICE)
        
        # Validate
        val_loss, val_c_index = evaluate(model, val_data, DEVICE)
        
        # Update scheduler
        scheduler.step(val_c_index)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_c_index'].append(val_c_index)
        history['learning_rate'].append(current_lr)
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val C-index: {val_c_index:.4f} | "
                  f"LR: {current_lr:.2e}")
        
        # Check for improvement
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_c_index': val_c_index,
                'val_loss': val_loss
            }, OUTPUT_DIR / "prodromal_prognostic_best.pth")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n✓ Early stopping at epoch {epoch}")
            print(f"  Best val C-index: {best_val_c_index:.4f} at epoch {best_epoch}")
            break
    
    # Training complete
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nBest validation C-index: {best_val_c_index:.4f} (epoch {best_epoch})")
    print(f"Total epochs: {len(history['train_loss'])}")
    
    # Load best model for final evaluation
    print(f"\n{'=' * 70}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'=' * 70}")
    
    checkpoint = torch.load(OUTPUT_DIR / "prodromal_prognostic_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_c_index = evaluate(model, test_data, DEVICE)
    
    print(f"\n✓ Test set performance:")
    print(f"  C-index: {test_c_index:.4f}")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Events: {split_info['test_events']}/{test_data.x.shape[0]}")
    
    # Compare to manifest PD
    print(f"\n{'=' * 70}")
    print("COMPARISON TO MANIFEST PD COHORT")
    print(f"{'=' * 70}")
    print(f"\n  Manifest PD test C-index: 0.38 [0.19, 0.69]")
    print(f"  Prodromal test C-index: {test_c_index:.4f}")
    
    if test_c_index >= 0.55:
        print(f"\n  ✓ TARGET MET (≥0.55)")
        print(f"    Evidence of generalizability to earlier disease stage!")
        print(f"    Framework adapts to real phenoconversion events.")
    elif test_c_index >= 0.45:
        print(f"\n  ~ MARGINAL IMPROVEMENT")
        print(f"    Better than manifest PD but below target.")
        print(f"    Investigate feature-level differences.")
    else:
        print(f"\n  ✗ BELOW MANIFEST PD")
        print(f"    Framework may not generalize to prodromal stage.")
        print(f"    Consider architecture refinements.")
    
    # Save results
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_architecture': {
            'num_features': NUM_FEATURES,
            'hidden_dim': HIDDEN_DIM,
            'num_gat_layers': NUM_GAT_LAYERS,
            'num_heads': NUM_HEADS,
            'dropout': DROPOUT,
            'total_params': n_params
        },
        'training': {
            'max_epochs': MAX_EPOCHS,
            'actual_epochs': len(history['train_loss']),
            'best_epoch': best_epoch,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'patience': PATIENCE
        },
        'data_splits': split_info,
        'performance': {
            'best_val_c_index': float(best_val_c_index),
            'test_c_index': float(test_c_index),
            'test_loss': float(test_loss)
        },
        'comparison': {
            'manifest_pd_test_c_index': 0.38,
            'manifest_pd_test_c_index_ci': [0.19, 0.69],
            'prodromal_test_c_index': float(test_c_index),
            'improvement': float(test_c_index - 0.38),
            'target_met': bool(test_c_index >= 0.55)
        }
    }
    
    with open(OUTPUT_DIR / "prodromal_test_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved prodromal_test_evaluation.json")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    axes[0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
    axes[0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cox PH Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # C-index curve
    axes[1].plot(history['val_c_index'], label='Val C-index', alpha=0.7, color='green')
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
    axes[1].axhline(0.55, color='orange', linestyle='--', alpha=0.5, label='Target (0.55)')
    axes[1].axhline(0.38, color='red', linestyle='--', alpha=0.5, label='Manifest PD (0.38)')
    axes[1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(best_val_c_index, color='green', linestyle=':', alpha=0.3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('C-index')
    axes[1].set_title('Validation C-index')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0.3, 1.0])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved training_curves.png")
    
    # Save training history
    history_df_data = {
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_c_index': history['val_c_index'],
        'learning_rate': history['learning_rate']
    }
    
    import pandas as pd
    history_df = pd.DataFrame(history_df_data)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"  ✓ Saved training_history.csv")
    
    print(f"\n{'=' * 70}")
    print("PHASE 8.1 TASK 12 COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\n✓ Model trained: {OUTPUT_DIR / 'prodromal_prognostic_best.pth'}")
    print(f"✓ Test C-index: {test_c_index:.4f}")
    print(f"✓ Target (≥0.55): {'MET ✓' if test_c_index >= 0.55 else 'NOT MET ✗'}")
    
    print(f"\nNext steps:")
    print(f"  1. Review training curves: {OUTPUT_DIR / 'training_curves.png'}")
    print(f"  2. Analyze test results: {OUTPUT_DIR / 'prodromal_test_evaluation.json'}")
    print(f"  3. Generate visualizations (Task 13)")
    print(f"  4. Write completion report (Task 14)")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
