"""
Training Pipeline for GIMAN-Conversion Model on Real PPMI Data.

This script implements the complete training pipeline for the GIMAN-Conversion
binary classification model using real PPMI cohort data (127 patients).

Features:
    - Weighted binary cross-entropy loss
    - AUC-ROC and AUC-PR evaluation
    - Early stopping with patience
    - Model checkpointing (best and last)
    - TensorBoard logging
    - Learning rate scheduling

Author: GIMAN Research Team
Date: 2025
Week: 3 - Training Implementation
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.giman_conversion import GIMANConversion, WeightedBCELoss
from src.utils.config_loader import load_config


class ClassificationMetrics:
    """
    Binary classification metrics for conversion prediction.
    
    Computes AUC-ROC and AUC-PR for model evaluation.
    
    AUC-ROC (Area Under ROC Curve):
        - Measures discrimination ability across all thresholds
        - Range: 0.5 (random) to 1.0 (perfect)
        - Robust to class imbalance
    
    AUC-PR (Area Under Precision-Recall Curve):
        - Focuses on positive class performance
        - Range: Baseline (% positives) to 1.0 (perfect)
        - More informative for imbalanced datasets
    
    Example:
        >>> probs = np.array([0.8, 0.3, 0.6])
        >>> labels = np.array([1, 0, 1])
        >>> metrics = ClassificationMetrics.compute(probs, labels)
        >>> print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        AUC-ROC: 1.000
    """
    
    @staticmethod
    def compute(
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            probs: Predicted probabilities [0, 1]
            labels: True binary labels {0, 1}
            
        Returns:
            Dictionary with AUC-ROC and AUC-PR
        """
        try:
            auc_roc = roc_auc_score(labels, probs)
        except ValueError:
            # Handle case where only one class present
            auc_roc = 0.5
        
        try:
            auc_pr = average_precision_score(labels, probs)
        except ValueError:
            # Handle case where only one class present
            auc_pr = labels.mean()  # Baseline
        
        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
        }


class GIMANConversionTrainer:
    """
    Trainer for GIMAN-Conversion model.
    
    Handles training loop, validation, early stopping, checkpointing,
    and logging for binary classification task.
    
    Args:
        config_path: Path to configuration YAML
        device: Device for training ('cuda' or 'cpu')
        
    Example:
        >>> trainer = GIMANConversionTrainer("configs/real_ppmi_dual_model.yaml")
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config_path: str = "configs/real_ppmi_dual_model.yaml",
        device: Optional[str] = None,
    ):
        # Load configuration
        self.config = load_config(config_path)
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Initialize model
        self.model = GIMANConversion(
            num_features=self.config.giman_conversion.model.num_features,
            hidden_dim=self.config.giman_conversion.model.hidden_dim,
            num_gat_layers=self.config.giman_conversion.model.num_gat_layers,
            num_heads=self.config.giman_conversion.model.num_heads,
            conversion_hidden_dims=self.config.giman_conversion.model.classifier_hidden_dims,
            dropout=self.config.giman_conversion.model.dropout,
        ).to(device)
        
        # Initialize loss
        self.criterion = WeightedBCELoss(
            pos_weight=self.config.giman_conversion.loss.pos_weight
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.giman_conversion.training.learning_rate,
            weight_decay=self.config.giman_conversion.training.weight_decay,
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize AUC-ROC
            factor=self.config.giman_conversion.scheduler.factor,
            patience=self.config.giman_conversion.scheduler.patience,
            min_lr=self.config.giman_conversion.scheduler.min_lr,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        
        # Setup logging
        self.setup_logging()
        
        # Load data
        self.load_data()
        
        print(f"[INIT] GIMAN-Conversion Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_logging(self):
        """Setup logging and TensorBoard."""
        # Create output directories
        self.output_dir = Path(self.config.experiment.output.conversion_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup TensorBoard
        if self.config.training.logging.tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir / "tensorboard")
        else:
            self.writer = None
    
    def load_data(self):
        """Load prepared training data."""
        data_dir = Path("data/02_processed/training_ready")
        
        self.logger.info("Loading training data...")
        self.train_data = torch.load(data_dir / "train_data.pt")
        self.val_data = torch.load(data_dir / "val_data.pt")
        self.test_data = torch.load(data_dir / "test_data.pt")
        
        self.logger.info(f"   Train: {self.train_data.num_nodes} patients")
        self.logger.info(f"   Val:   {self.val_data.num_nodes} patients")
        self.logger.info(f"   Test:  {self.test_data.num_nodes} patients")
        
        # Load hybrid real+simulated conversion labels
        self.load_hybrid_conversion_labels()
    
    def load_hybrid_conversion_labels(self):
        """
        Load hybrid real+simulated conversion labels.
        
        This method loads conversion data that combines:
        - Real PPMI phenoconversion events (6 patients, 4.7%)
        - Clinically-informed simulated conversions (38 patients, 29.9%)
        - Total conversion rate: ~35% for adequate statistical power
        
        Conversion criteria based on:
        - Motor symptom progression (NP3TOT increase + H&Y worsening)
        - Cognitive decline (MoCA decrease)
        - Functional impairment emergence
        """
        import pandas as pd
        
        self.logger.info("Loading hybrid conversion labels...")
        
        # Load hybrid conversion data
        conversion_file = Path("data/02_processed/conversion_labels_hybrid.csv")
        conversion_df = pd.read_csv(conversion_file)
        
        # Load patient split assignments
        split_file = Path("data/02_processed/training_ready/split_info.json")
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        
        # Extract labels for each split
        train_patnos = split_info['train_patnos']
        val_patnos = split_info['val_patnos']
        test_patnos = split_info['test_patnos']
        
        # Filter conversion data by split
        train_conv = conversion_df[conversion_df['PATNO'].isin(train_patnos)]
        val_conv = conversion_df[conversion_df['PATNO'].isin(val_patnos)]
        test_conv = conversion_df[conversion_df['PATNO'].isin(test_patnos)]
        
        # Convert to tensors
        self.train_labels = torch.tensor(
            train_conv['converted'].values,
            dtype=torch.float32
        )
        
        self.val_labels = torch.tensor(
            val_conv['converted'].values,
            dtype=torch.float32
        )
        
        self.test_labels = torch.tensor(
            test_conv['converted'].values,
            dtype=torch.float32
        )
        
        # Log statistics
        train_rate = self.train_labels.mean().item()
        val_rate = self.val_labels.mean().item()
        test_rate = self.test_labels.mean().item()
        
        self.logger.info(f"   Train conversion: {self.train_labels.sum():.0f}/{len(self.train_labels)} ({train_rate:.1%})")
        self.logger.info(f"   Val conversion:   {self.val_labels.sum():.0f}/{len(self.val_labels)} ({val_rate:.1%})")
        self.logger.info(f"   Test conversion:  {self.test_labels.sum():.0f}/{len(self.test_labels)} ({test_rate:.1%})")
        
        # Count real vs simulated converters (real converters have conversion_type != 'simulated_rapid_progression')
        train_real = train_conv[
            (train_conv['converted'] == 1) & 
            (train_conv['conversion_type'] != 'simulated_rapid_progression')
        ].shape[0]
        train_sim = train_conv[
            (train_conv['converted'] == 1) & 
            (train_conv['conversion_type'] == 'simulated_rapid_progression')
        ].shape[0]
        
        self.logger.info(f"   Train composition: {train_real} real + {train_sim} simulated converters")
        self.logger.info("   ✅ Using hybrid real+simulated conversion labels")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Move data to device
        train_data = self.train_data.to(self.device)
        labels = self.train_labels.to(self.device)
        
        # Forward pass
        logits = self.model(train_data.x, train_data.edge_index)
        probs = torch.sigmoid(logits)
        
        # Compute loss
        loss = self.criterion(logits.squeeze(), labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.training.gradient_clipping.enabled:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clipping.max_norm
            )
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            probs_np = probs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy()
            
            metrics = ClassificationMetrics.compute(probs_np, labels_np)
        
        return {
            'loss': loss.item(),
            'auc_roc': metrics['auc_roc'],
            'auc_pr': metrics['auc_pr'],
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            val_data = self.val_data.to(self.device)
            labels = self.val_labels.to(self.device)
            
            # Forward pass
            logits = self.model(val_data.x, val_data.edge_index)
            probs = torch.sigmoid(logits)
            
            # Compute loss
            loss = self.criterion(logits.squeeze(), labels)
            
            # Compute metrics
            probs_np = probs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy()
            
            metrics = ClassificationMetrics.compute(probs_np, labels_np)
        
        return {
            'loss': loss.item(),
            'auc_roc': metrics['auc_roc'],
            'auc_pr': metrics['auc_pr'],
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'config': self.config.to_dict(),
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"   💾 Saved best checkpoint (AUC-ROC: {self.best_val_auc:.4f})")
    
    def train(self):
        """Execute complete training loop."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("GIMAN-CONVERSION TRAINING")
        self.logger.info("=" * 70)
        
        max_epochs = self.config.giman_conversion.training.max_epochs
        patience = self.config.giman_conversion.training.early_stopping.patience
        
        self.logger.info(f"Max epochs: {max_epochs}")
        self.logger.info(f"Early stopping patience: {patience}")
        self.logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(1, max_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['auc_roc'])
            
            # Logging
            epoch_time = time.time() - epoch_start
            
            self.logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train AUC: {train_metrics['auc_roc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('AUC_ROC/train', train_metrics['auc_roc'], epoch)
                self.writer.add_scalar('AUC_ROC/val', val_metrics['auc_roc'], epoch)
                self.writer.add_scalar('AUC_PR/train', train_metrics['auc_pr'], epoch)
                self.writer.add_scalar('AUC_PR/val', val_metrics['auc_pr'], epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for improvement
            if val_metrics['auc_roc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc_roc']
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Save last checkpoint periodically
            if epoch % self.config.training.checkpoint.save_frequency == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                self.logger.info(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                self.logger.info(f"   Best validation AUC-ROC: {self.best_val_auc:.4f}")
                break
        
        total_time = time.time() - start_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("✅ TRAINING COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Best validation AUC-ROC: {self.best_val_auc:.4f}")
        self.logger.info(f"Final model saved to: {self.checkpoint_dir}")
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        # Save training summary
        self.save_training_summary(total_time)
    
    def save_training_summary(self, total_time: float):
        """Save training summary to JSON."""
        summary = {
            'model': 'GIMAN-Conversion',
            'total_epochs': self.current_epoch,
            'best_val_auc_roc': float(self.best_val_auc),
            'total_training_time_minutes': total_time / 60,
            'device': self.device,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'config': self.config.to_dict(),
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved to: {summary_path}")


def main():
    """Main training execution."""
    print("\n" + "=" * 70)
    print("GIMAN-CONVERSION TRAINING ON REAL PPMI DATA")
    print("=" * 70)
    print("Week 3: Training Implementation")
    pr