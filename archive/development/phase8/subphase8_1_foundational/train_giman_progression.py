"""
GIMAN-Progression Training Script.

This script implements the complete training pipeline for GIMAN-Progression,
including training loop, validation, early stopping, checkpointing, and logging.

Features:
    - Cox partial likelihood loss optimization
    - Concordance index (C-index) evaluation
    - Early stopping with patience
    - Model checkpointing (best and last)
    - TensorBoard logging
    - Learning rate scheduling

Author: GIMAN Development Team
Date: October 8, 2025
Version: 8.1.0 - Week 2 Training
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from models.giman_progression import GIMANProgression, CoxPartialLikelihoodLoss
from src.utils.config_loader import load_config


class ConcordanceIndex:
    """
    Concordance Index (C-index) metric for survival analysis.
    
    The C-index measures the model's ability to correctly rank patients
    by their risk scores. It's equivalent to AUC for survival analysis.
    
    C-index = P(risk_i > risk_j | time_i < time_j and event_i = 1)
    """
    
    @staticmethod
    def compute(
        risk_scores: np.ndarray,
        event_times: np.ndarray,
        event_observed: np.ndarray,
    ) -> float:
        """
        Compute concordance index.
        
        Args:
            risk_scores: Predicted risk scores [num_patients]
            event_times: Observed event times [num_patients]
            event_observed: Event indicators (1=event, 0=censored) [num_patients]
            
        Returns:
            C-index value (0.5 = random, 1.0 = perfect)
        """
        n = len(risk_scores)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            if event_observed[i] == 0:
                continue  # Skip censored events
            
            for j in range(n):
                if i == j:
                    continue
                
                # Only compare if j's time is longer than i's
                if event_times[j] > event_times[i]:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
        
        if concordant + discordant == 0:
            return 0.5
        
        return concordant / (concordant + discordant)


class GIMANProgressionTrainer:
    """
    Trainer for GIMAN-Progression model.
    
    Handles training loop, validation, early stopping, checkpointing,
    and logging for survival analysis task.
    
    Args:
        config_path: Path to configuration YAML
        device: Device for training ('cuda' or 'cpu')
        
    Example:
        >>> trainer = GIMANProgressionTrainer("configs/real_ppmi_dual_model.yaml")
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
        self.model = GIMANProgression(
            num_features=self.config.giman_progression.model.num_features,
            hidden_dim=self.config.giman_progression.model.hidden_dim,
            num_gat_layers=self.config.giman_progression.model.num_gat_layers,
            num_heads=self.config.giman_progression.model.num_heads,
            survival_hidden_dims=self.config.giman_progression.model.survival_hidden_dims,
            dropout=self.config.giman_progression.model.dropout,
        ).to(device)
        
        # Initialize loss
        self.criterion = CoxPartialLikelihoodLoss()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.giman_progression.training.learning_rate,
            weight_decay=self.config.giman_progression.training.weight_decay,
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize C-index
            factor=self.config.giman_progression.scheduler.factor,
            patience=self.config.giman_progression.scheduler.patience,
            min_lr=self.config.giman_progression.scheduler.min_lr,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_cindex = 0.0
        self.epochs_without_improvement = 0
        
        # Setup logging
        self.setup_logging()
        
        # Load data
        self.load_data()
        
        print(f"[INIT] GIMAN-Progression Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_logging(self):
        """Setup logging and TensorBoard."""
        # Create output directories
        self.output_dir = Path(self.config.experiment.output.progression_dir)
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
        """Load prepared training data and real survival labels."""
        data_dir = Path("data/02_processed/training_ready")
        
        self.logger.info("Loading training data...")
        self.train_data = torch.load(data_dir / "train_data.pt", weights_only=False)
        self.val_data = torch.load(data_dir / "val_data.pt", weights_only=False)
        self.test_data = torch.load(data_dir / "test_data.pt", weights_only=False)
        
        self.logger.info(f"   Train: {self.train_data.num_nodes} patients")
        self.logger.info(f"   Val:   {self.val_data.num_nodes} patients")
        self.logger.info(f"   Test:  {self.test_data.num_nodes} patients")
        self._load_real_survival_labels(data_dir)

    def _load_real_survival_labels(self, data_dir: Path):
        """Load real survival labels and align by split PATNOs."""
        split_info_path = data_dir / "split_info.json"
        survival_path = Path("data/02_processed/progression_survival_data.csv")

        if not split_info_path.exists():
            raise FileNotFoundError(
                f"Missing split_info.json required for real labels: {split_info_path}"
            )
        if not survival_path.exists():
            raise FileNotFoundError(
                f"Missing real survival labels file: {survival_path}"
            )

        with open(split_info_path) as f:
            split_info = json.load(f)

        survival_df = pd.read_csv(survival_path)
        if {"event_time", "event_observed"}.issubset(survival_df.columns):
            time_col, event_col = "event_time", "event_observed"
        elif {"time_to_event", "phenoconverted"}.issubset(survival_df.columns):
            time_col, event_col = "time_to_event", "phenoconverted"
        else:
            raise ValueError(
                "Real survival labels must include either "
                "('event_time','event_observed') or ('time_to_event','phenoconverted')."
            )

        survival_df["PATNO"] = pd.to_numeric(survival_df["PATNO"], errors="coerce")
        survival_df = survival_df.dropna(subset=["PATNO"])
        survival_df["PATNO"] = survival_df["PATNO"].astype(int)
        survival_df[time_col] = pd.to_numeric(
            survival_df[time_col], errors="coerce"
        )
        survival_df[event_col] = pd.to_numeric(
            survival_df[event_col], errors="coerce"
        )
        survival_df = survival_df.dropna(subset=[time_col, event_col])

        label_lookup = (
            survival_df.set_index("PATNO")[[time_col, event_col]].to_dict("index")
        )

        def build_split_labels(split_name: str, expected_size: int):
            patnos = split_info.get(f"{split_name}_patnos")
            if patnos is None:
                raise ValueError(f"Missing {split_name}_patnos in split_info.json")

            missing_patnos = [p for p in patnos if int(p) not in label_lookup]
            if missing_patnos:
                raise ValueError(
                    f"Missing real labels for {split_name} split PATNOs: "
                    f"{missing_patnos[:10]}{'...' if len(missing_patnos) > 10 else ''}"
                )

            event_time = torch.tensor(
                [float(label_lookup[int(p)][time_col]) for p in patnos],
                dtype=torch.float32,
            )
            event_observed = torch.tensor(
                [float(label_lookup[int(p)][event_col]) for p in patnos],
                dtype=torch.float32,
            )

            if len(event_time) != expected_size:
                raise ValueError(
                    f"{split_name} label size mismatch: "
                    f"labels={len(event_time)} vs graph_nodes={expected_size}"
                )

            if (event_time < 0).any():
                raise ValueError(f"{split_name} contains negative event_time values")

            return event_time, event_observed

        self.train_event_times, self.train_event_observed = build_split_labels(
            "train", self.train_data.num_nodes
        )
        self.val_event_times, self.val_event_observed = build_split_labels(
            "val", self.val_data.num_nodes
        )
        self.test_event_times, self.test_event_observed = build_split_labels(
            "test", self.test_data.num_nodes
        )

        self.logger.info("Loaded real survival labels from data/02_processed/progression_survival_data.csv")
        self.logger.info(
            f"   Event rates | Train: {self.train_event_observed.mean():.4f}, "
            f"Val: {self.val_event_observed.mean():.4f}, "
            f"Test: {self.test_event_observed.mean():.4f}"
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Move data to device
        train_data = self.train_data.to(self.device)
        event_times = self.train_event_times.to(self.device)
        event_observed = self.train_event_observed.to(self.device)
        
        # Forward pass
        risk_scores = self.model(train_data.x, train_data.edge_index)
        
        # Compute loss
        loss = self.criterion(risk_scores, event_times, event_observed)
        
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
        
        # Compute C-index
        with torch.no_grad():
            risk_scores_np = risk_scores.cpu().numpy().flatten()
            event_times_np = event_times.cpu().numpy()
            event_observed_np = event_observed.cpu().numpy()
            
            train_cindex = ConcordanceIndex.compute(
                risk_scores_np, event_times_np, event_observed_np
            )
        
        return {
            'loss': loss.item(),
            'cindex': train_cindex,
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
            event_times = self.val_event_times.to(self.device)
            event_observed = self.val_event_observed.to(self.device)
            
            # Forward pass
            risk_scores = self.model(val_data.x, val_data.edge_index)
            
            # Compute loss
            loss = self.criterion(risk_scores, event_times, event_observed)
            
            # Compute C-index
            risk_scores_np = risk_scores.cpu().numpy().flatten()
            event_times_np = event_times.cpu().numpy()
            event_observed_np = event_observed.cpu().numpy()
            
            val_cindex = ConcordanceIndex.compute(
                risk_scores_np, event_times_np, event_observed_np
            )
        
        return {
            'loss': loss.item(),
            'cindex': val_cindex,
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_cindex': self.best_val_cindex,
            'config': self.config.to_dict(),
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"   💾 Saved best checkpoint (C-index: {self.best_val_cindex:.4f})")
    
    def train(self):
        """Execute complete training loop."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("GIMAN-PROGRESSION TRAINING")
        self.logger.info("=" * 70)
        
        max_epochs = self.config.giman_progression.training.max_epochs
        patience = self.config.giman_progression.training.early_stopping.patience
        
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
            self.scheduler.step(val_metrics['cindex'])
            
            # Logging
            epoch_time = time.time() - epoch_start
            
            self.logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train C-idx: {train_metrics['cindex']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val C-idx: {val_metrics['cindex']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('CIndex/train', train_metrics['cindex'], epoch)
                self.writer.add_scalar('CIndex/val', val_metrics['cindex'], epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for improvement
            if val_metrics['cindex'] > self.best_val_cindex:
                self.best_val_cindex = val_metrics['cindex']
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
                self.logger.info(f"   Best validation C-index: {self.best_val_cindex:.4f}")
                break
        
        total_time = time.time() - start_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("✅ TRAINING COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Best validation C-index: {self.best_val_cindex:.4f}")
        self.logger.info(f"Final model saved to: {self.checkpoint_dir}")
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        # Save training summary
        self.save_training_summary(total_time)
    
    def save_training_summary(self, total_time: float):
        """Save training summary to JSON."""
        summary = {
            'model': 'GIMAN-Progression',
            'total_epochs': self.current_epoch,
            'best_val_cindex': float(self.best_val_cindex),
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
    trainer = GIMANProgressionTrainer(
        config_path="configs/real_ppmi_dual_model.yaml"
    )
    trainer.train()


if __name__ == '__main__':
    main()
