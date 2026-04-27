"""
Train Heterogeneity VAE on GIMAN GAT embeddings.

This script trains the adapted VaDER VAE architecture on 128-dim embeddings
extracted from Phase 8.2 GIMAN-Progression model. Tests multiple latent
dimensions to find optimal compression-reconstruction trade-off.

Training strategy:
- Patient-level train/val/test split (70/15/15)
- AdamW optimizer with ReduceLROnPlateau scheduler
- Early stopping based on validation reconstruction loss
- Monitor reconstruction loss, KL divergence, and total loss
- Target: reconstruction loss < 0.1

Author: GIMAN Research Team
Date: October 14, 2025
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add models directory to path
models_path = Path(__file__).resolve().parent.parent / "models"
sys.path.insert(0, str(models_path))

from heterogeneity_vae import (
    build_heterogeneity_vae,
    compute_vae_loss,
    HeterogeneityVAE
)


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for GIMAN embeddings."""
    
    def __init__(self, embeddings: np.ndarray, patient_ids: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.patient_ids = patient_ids
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.patient_ids[idx]


class VAETrainer:
    """Train and evaluate Heterogeneity VAE."""
    
    def __init__(
        self,
        latent_dim: int = 16,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        beta: float = 1.0,
        max_epochs: int = 200,
        early_stopping_patience: int = 20,
        device: str = "cpu"
    ):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device)
        
        # Paths
        self.project_root = Path(__file__).resolve().parents[5]
        self.embeddings_path = self.project_root / "data" / "05_embeddings" / "giman_gat_embeddings.csv"
        self.output_dir = self.project_root / "archive" / "development" / "phase8" / "subphase8_4_vae_heterogeneity" / "results"
        self.checkpoint_dir = self.project_root / "archive" / "development" / "phase8" / "subphase8_4_vae_heterogeneity" / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and data
        self.model = None
        self.scaler = StandardScaler()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def load_and_prepare_data(self) -> None:
        """Load embeddings and prepare train/val/test splits."""
        print("=" * 80)
        print("LOADING AND PREPARING DATA")
        print("=" * 80)
        
        # Load embeddings
        df = pd.read_csv(self.embeddings_path)
        print(f"\nLoaded: {len(df)} observations from {df['PATNO'].nunique()} patients")
        
        # Extract embedding columns
        embedding_cols = [col for col in df.columns if col.startswith('EMB_')]
        X = df[embedding_cols].values
        patient_ids = df['PATNO'].values
        
        print(f"Embedding dimension: {X.shape[1]}")
        
        # Patient-level train/val/test split (70/15/15)
        unique_patients = df['PATNO'].unique()
        train_patients, test_patients = train_test_split(
            unique_patients, test_size=0.15, random_state=42
        )
        train_patients, val_patients = train_test_split(
            train_patients, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
        )
        
        # Create masks
        train_mask = np.isin(patient_ids, train_patients)
        val_mask = np.isin(patient_ids, val_patients)
        test_mask = np.isin(patient_ids, test_patients)
        
        print(f"\nSplit:")
        print(f"  Train: {train_mask.sum()} obs from {len(train_patients)} patients ({len(train_patients)/len(unique_patients)*100:.1f}%)")
        print(f"  Val:   {val_mask.sum()} obs from {len(val_patients)} patients ({len(val_patients)/len(unique_patients)*100:.1f}%)")
        print(f"  Test:  {test_mask.sum()} obs from {len(test_patients)} patients ({len(test_patients)/len(unique_patients)*100:.1f}%)")
        
        # Standardize embeddings (fit on train only)
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        
        print(f"\nStandardizing embeddings...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"  Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
        print(f"  Val mean: {X_val_scaled.mean():.4f}, std: {X_val_scaled.std():.4f}")
        print(f"  Test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")
        
        # Create datasets and dataloaders
        train_dataset = EmbeddingDataset(X_train_scaled, patient_ids[train_mask])
        val_dataset = EmbeddingDataset(X_val_scaled, patient_ids[val_mask])
        test_dataset = EmbeddingDataset(X_test_scaled, patient_ids[test_mask])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"\nDataLoaders created:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
    
    def build_model(self) -> None:
        """Build and initialize VAE model."""
        print("\n" + "=" * 80)
        print(f"BUILDING VAE MODEL (latent_dim={self.latent_dim})")
        print("=" * 80)
        
        self.model = build_heterogeneity_vae(
            input_dim=128,
            latent_dim=self.latent_dim,
            dropout=0.3,
            device=str(self.device)
        )
        
        print(f"\nModel architecture:")
        print(self.model)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
    
    def train_epoch(self, optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        n_batches = 0
        
        for embeddings, _ in self.train_loader:
            embeddings = embeddings.to(self.device)
            
            # Forward pass
            x_recon, mu, logvar, z = self.model(embeddings)
            
            # Compute loss
            loss, recon_loss, kl_loss = compute_vae_loss(
                x_recon, embeddings, mu, logvar, self.beta, return_components=True
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            n_batches += 1
        
        # Average over batches
        return {
            'loss': epoch_loss / n_batches,
            'recon_loss': epoch_recon_loss / n_batches,
            'kl_loss': epoch_kl_loss / n_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation or test set."""
        self.model.eval()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for embeddings, _ in dataloader:
                embeddings = embeddings.to(self.device)
                
                # Forward pass
                x_recon, mu, logvar, z = self.model(embeddings)
                
                # Compute loss
                loss, recon_loss, kl_loss = compute_vae_loss(
                    x_recon, embeddings, mu, logvar, self.beta, return_components=True
                )
                
                # Accumulate
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                n_batches += 1
        
        # Average over batches
        return {
            'loss': epoch_loss / n_batches,
            'recon_loss': epoch_recon_loss / n_batches,
            'kl_loss': epoch_kl_loss / n_batches
        }
    
    def train(self) -> None:
        """Full training loop with early stopping."""
        print("\n" + "=" * 80)
        print("TRAINING VAE")
        print("=" * 80)
        print(f"\nHyperparameters:")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Beta (KL weight): {self.beta}")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Early stopping patience: {self.early_stopping_patience}")
        print(f"  Device: {self.device}")
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(1, self.max_epochs + 1):
            # Train
            train_metrics = self.train_epoch(optimizer)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            
            # Scheduler step
            scheduler.step(val_metrics['recon_loss'])
            
            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{self.max_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
                      f"Recon: {train_metrics['recon_loss']:.4f} | "
                      f"KL: {train_metrics['kl_loss']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
                      f"Recon: {val_metrics['recon_loss']:.4f} | "
                      f"KL: {val_metrics['kl_loss']:.4f}")
            
            # Early stopping check
            if val_metrics['recon_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['recon_loss']
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                
                if epoch % 5 == 0 or epoch == 1:
                    print(f"  ✅ New best val recon loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                    print(f"   Best val recon loss: {self.best_val_loss:.4f}")
                    break
        
        training_time = time.time() - start_time
        print(f"\n✅ Training complete in {training_time/60:.1f} minutes")
        print(f"   Best val recon loss: {self.best_val_loss:.4f}")
    
    def test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        print("\n" + "=" * 80)
        print("EVALUATING ON TEST SET")
        print("=" * 80)
        
        # Load best model
        best_checkpoint_path = self.checkpoint_dir / f"best_vae_latent{self.latent_dim}.pth"
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_metrics = self.evaluate(self.test_loader)
        
        print(f"\nTest Results:")
        print(f"  Total loss: {test_metrics['loss']:.4f}")
        print(f"  Reconstruction loss: {test_metrics['recon_loss']:.4f}")
        print(f"  KL divergence: {test_metrics['kl_loss']:.4f}")
        
        return test_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'latent_dim': self.latent_dim,
            'val_metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }
        
        if is_best:
            path = self.checkpoint_dir / f"best_vae_latent{self.latent_dim}.pth"
            torch.save(checkpoint, path)
    
    def plot_training_history(self) -> None:
        """Plot training curves."""
        print("\n" + "=" * 80)
        print("GENERATING TRAINING PLOTS")
        print("=" * 80)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'VAE Training History (latent_dim={self.latent_dim})', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Total loss
        ax = axes[0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train', alpha=0.7)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reconstruction loss
        ax = axes[1]
        ax.plot(epochs, self.history['train_recon_loss'], 'b-', label='Train', alpha=0.7)
        ax.plot(epochs, self.history['val_recon_loss'], 'r-', label='Val', alpha=0.7)
        ax.axhline(y=0.1, color='g', linestyle='--', label='Target (0.1)', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title('Reconstruction Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # KL divergence
        ax = axes[2]
        ax.plot(epochs, self.history['train_kl_loss'], 'b-', label='Train', alpha=0.7)
        ax.plot(epochs, self.history['val_kl_loss'], 'r-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f"training_history_latent{self.latent_dim}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")
        plt.close()
    
    def save_results(self, test_metrics: Dict) -> None:
        """Save training results to JSON."""
        results = {
            'latent_dim': self.latent_dim,
            'hyperparameters': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'beta': self.beta,
                'max_epochs': self.max_epochs,
                'early_stopping_patience': self.early_stopping_patience
            },
            'best_val_recon_loss': self.best_val_loss,
            'test_metrics': test_metrics,
            'training_history': self.history,
            'total_epochs': len(self.history['train_loss'])
        }
        
        save_path = self.output_dir / f"results_latent{self.latent_dim}.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Saved results: {save_path}")


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("PHASE 8.4: VAE TRAINING ON GIMAN EMBEDDINGS")
    print("=" * 80)
    print("\nTraining Heterogeneity VAE with multiple latent dimensions\n")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Test multiple latent dimensions
    latent_dims = [8, 12, 16]
    
    all_results = {}
    
    for latent_dim in latent_dims:
        print("\n" + "=" * 80)
        print(f"TRAINING WITH LATENT_DIM = {latent_dim}")
        print("=" * 80)
        
        # Initialize trainer
        trainer = VAETrainer(
            latent_dim=latent_dim,
            batch_size=64,
            learning_rate=0.001,
            beta=1.0,
            max_epochs=200,
            early_stopping_patience=20,
            device=device
        )
        
        # Load data (only once)
        if latent_dim == latent_dims[0]:
            trainer.load_and_prepare_data()
            # Save loaders for next iterations
            train_loader = trainer.train_loader
            val_loader = trainer.val_loader
            test_loader = trainer.test_loader
            scaler = trainer.scaler
        else:
            # Reuse loaders
            trainer.train_loader = train_loader
            trainer.val_loader = val_loader
            trainer.test_loader = test_loader
            trainer.scaler = scaler
        
        # Build model
        trainer.build_model()
        
        # Train
        trainer.train()
        
        # Test
        test_metrics = trainer.test()
        
        # Plot and save
        trainer.plot_training_history()
        trainer.save_results(test_metrics)
        
        # Store results
        all_results[latent_dim] = {
            'best_val_recon_loss': trainer.best_val_loss,
            'test_recon_loss': test_metrics['recon_loss'],
            'test_kl_loss': test_metrics['kl_loss']
        }
        
        print(f"\n✅ Completed latent_dim={latent_dim}")
    
    # Compare results
    print("\n" + "=" * 80)
    print("FINAL COMPARISON ACROSS LATENT DIMENSIONS")
    print("=" * 80)
    
    print(f"\n{'Latent Dim':<12} {'Val Recon Loss':<18} {'Test Recon Loss':<18} {'Test KL Loss':<15}")
    print("-" * 80)
    for latent_dim, results in all_results.items():
        print(f"{latent_dim:<12} {results['best_val_recon_loss']:<18.4f} "
              f"{results['test_recon_loss']:<18.4f} {results['test_kl_loss']:<15.4f}")
    
    # Find best
    best_latent_dim = min(all_results.keys(), key=lambda k: all_results[k]['test_recon_loss'])
    print(f"\n✅ Best latent dimension: {best_latent_dim} "
          f"(test recon loss: {all_results[best_latent_dim]['test_recon_loss']:.4f})")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review training curves and select best latent dimension")
    print("  2. Extract latent codes for all patients using best model")
    print("  3. Perform biological correlation analysis")
    print("  4. Compare to Phase 4 discrete subtypes")


if __name__ == "__main__":
    main()
