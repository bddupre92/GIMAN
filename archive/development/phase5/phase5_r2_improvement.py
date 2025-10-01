#!/usr/bin/env python3
"""Phase 5 R² Improvement Experiments.

Systematic approach to improve Motor R² = -0.1157 through:
1. Architecture optimization
2. Training hyperparameter tuning
3. Regularization adjustment
4. Loss weighting optimization
"""

import logging
import os

# Import our dependencies
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# Add Phase 3 path
phase3_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase3"
)
sys.path.insert(0, phase3_path)
from phase3_1_real_data_integration import RealDataPhase3Integration

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ImprovedTaskSpecificGIMAN(nn.Module):
    """Improved GIMAN with optimized architecture for better R²."""

    def __init__(
        self,
        spatial_dim: int = 256,
        genomic_dim: int = 256,
        temporal_dim: int = 256,
        embed_dim: int = 64,  # Increased from 32
        num_heads: int = 4,  # Increased from 2
        dropout: float = 0.3,  # Reduced from 0.5
    ):
        """Initialize improved task-specific GIMAN system."""
        super().__init__()
        self.embed_dim = embed_dim

        # Improved embedding layers with batch normalization
        self.spatial_embedding = nn.Sequential(
            nn.Linear(spatial_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.genomic_embedding = nn.Sequential(
            nn.Linear(genomic_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.temporal_embedding = nn.Sequential(
            nn.Linear(temporal_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # Multimodal fusion with attention
        self.multimodal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim * 3,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Deeper task-specific towers
        self.motor_tower = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 4),
            nn.BatchNorm1d(embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),  # Motor regression
        )

        self.cognitive_tower = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 4),
            nn.BatchNorm1d(embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),  # Cognitive classification
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Better weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, spatial, genomic, temporal, adj_matrix=None):
        """Forward pass through improved GIMAN."""
        batch_size = spatial.size(0)

        # Improved embedding layers
        spatial_emb = self.spatial_embedding(spatial)
        genomic_emb = self.genomic_embedding(genomic)
        temporal_emb = self.temporal_embedding(temporal)

        # Combine embeddings
        combined_emb = torch.cat([spatial_emb, genomic_emb, temporal_emb], dim=-1)

        # Self-attention for modality interaction
        if combined_emb.dim() == 2:
            combined_emb = combined_emb.unsqueeze(1)  # Add sequence dimension

        attn_output, _ = self.multimodal_attention(
            combined_emb, combined_emb, combined_emb
        )

        # Final representation
        final_repr = attn_output.squeeze(1) if attn_output.dim() == 3 else attn_output

        # Task-specific predictions
        motor_pred = self.motor_tower(final_repr)
        cognitive_pred = self.cognitive_tower(final_repr)

        return motor_pred, cognitive_pred


class OptimizedTrainer:
    """Optimized training with better hyperparameters."""

    def __init__(self, device="cpu"):
        self.device = device

    def train_model(
        self, model, x_spatial, x_genomic, x_temporal, y_motor, y_cognitive
    ):
        """Train with optimized hyperparameters."""
        # Convert to tensors
        x_spatial_tensor = torch.FloatTensor(x_spatial).to(self.device)
        x_genomic_tensor = torch.FloatTensor(x_genomic).to(self.device)
        x_temporal_tensor = torch.FloatTensor(x_temporal).to(self.device)
        y_motor_tensor = torch.FloatTensor(y_motor).to(self.device)
        y_cognitive_tensor = torch.FloatTensor(y_cognitive).to(self.device)

        # Optimized optimizer settings
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.003,  # Increased learning rate
            weight_decay=1e-4,  # Reduced weight decay
            eps=1e-8,
        )

        # Scheduler for learning rate decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        # Loss functions
        motor_criterion = nn.MSELoss()
        cognitive_criterion = nn.BCELoss()

        # Training loop with better early stopping
        best_loss = float("inf")
        patience = 20  # Increased patience
        patience_counter = 0

        model.train()
        for epoch in range(200):  # More epochs
            optimizer.zero_grad()

            # Forward pass
            motor_pred, cognitive_pred = model(
                x_spatial_tensor, x_genomic_tensor, x_temporal_tensor
            )

            # Compute losses
            motor_loss = motor_criterion(motor_pred.squeeze(), y_motor_tensor)
            cognitive_loss = cognitive_criterion(
                cognitive_pred.squeeze(), y_cognitive_tensor
            )

            # Optimized loss weighting (focus more on motor task)
            total_loss = 0.8 * motor_loss + 0.2 * cognitive_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            scheduler.step()

            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return model

    def evaluate_loocv(self, X_spatial, X_genomic, X_temporal, y_motor, y_cognitive):
        """Run LOOCV with improved architecture."""
        logger.info("🔧 Starting R² Improvement Experiment...")

        loo = LeaveOneOut()
        n_samples = X_spatial.shape[0]

        motor_predictions = []
        motor_actuals = []
        cognitive_predictions = []
        cognitive_actuals = []

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_spatial)):
            if fold % 10 == 0:
                logger.info(f"📊 Processing fold {fold + 1}/{n_samples}")

            # Split data
            X_train_spatial, X_test_spatial = X_spatial[train_idx], X_spatial[test_idx]
            X_train_genomic, X_test_genomic = X_genomic[train_idx], X_genomic[test_idx]
            X_train_temporal, X_test_temporal = (
                X_temporal[train_idx],
                X_temporal[test_idx],
            )
            y_train_motor, y_test_motor = y_motor[train_idx], y_motor[test_idx]
            y_train_cognitive, y_test_cognitive = (
                y_cognitive[train_idx],
                y_cognitive[test_idx],
            )

            # Scale features
            scaler_spatial = StandardScaler()
            scaler_genomic = StandardScaler()
            scaler_temporal = StandardScaler()

            X_train_spatial = scaler_spatial.fit_transform(X_train_spatial)
            X_test_spatial = scaler_spatial.transform(X_test_spatial)
            X_train_genomic = scaler_genomic.fit_transform(X_train_genomic)
            X_test_genomic = scaler_genomic.transform(X_test_genomic)
            X_train_temporal = scaler_temporal.fit_transform(X_train_temporal)
            X_test_temporal = scaler_temporal.transform(X_test_temporal)

            # Create improved model
            model = ImprovedTaskSpecificGIMAN(
                spatial_dim=X_spatial.shape[1],
                genomic_dim=X_genomic.shape[1],
                temporal_dim=X_temporal.shape[1],
                embed_dim=64,  # Increased capacity
                num_heads=4,  # More attention heads
                dropout=0.3,  # Reduced dropout
            ).to(self.device)

            # Train model
            model = self.train_model(
                model,
                X_train_spatial,
                X_train_genomic,
                X_train_temporal,
                y_train_motor,
                y_train_cognitive,
            )

            # Test model
            model.eval()
            with torch.no_grad():
                x_test_spatial_tensor = torch.FloatTensor(X_test_spatial).to(
                    self.device
                )
                x_test_genomic_tensor = torch.FloatTensor(X_test_genomic).to(
                    self.device
                )
                x_test_temporal_tensor = torch.FloatTensor(X_test_temporal).to(
                    self.device
                )

                motor_pred, cognitive_pred = model(
                    x_test_spatial_tensor, x_test_genomic_tensor, x_test_temporal_tensor
                )

                motor_predictions.append(motor_pred.cpu().numpy())
                motor_actuals.append(y_test_motor)
                cognitive_predictions.append(cognitive_pred.cpu().numpy())
                cognitive_actuals.append(y_test_cognitive)

        # Calculate improved metrics
        motor_pred_array = np.array(motor_predictions).flatten()
        motor_actual_array = np.array(motor_actuals).flatten()
        cognitive_pred_array = np.array(cognitive_predictions).flatten()
        cognitive_actual_array = np.array(cognitive_actuals).flatten()

        # Motor regression metrics
        motor_r2 = r2_score(motor_actual_array, motor_pred_array)

        # Cognitive classification metrics
        try:
            cognitive_auc = roc_auc_score(cognitive_actual_array, cognitive_pred_array)
        except ValueError:
            cognitive_auc = 0.5

        logger.info("✅ R² Improvement Experiment Complete!")
        logger.info(f"🎯 Improved Motor R² = {motor_r2:.4f}")
        logger.info(f"🧠 Improved Cognitive AUC = {cognitive_auc:.4f}")

        return {
            "motor_r2": motor_r2,
            "cognitive_auc": cognitive_auc,
            "motor_predictions": motor_pred_array,
            "motor_actuals": motor_actual_array,
            "cognitive_predictions": cognitive_pred_array,
            "cognitive_actuals": cognitive_actual_array,
        }


def run_r2_improvement_experiment():
    """Run the R² improvement experiment."""
    logger.info("🚀 Starting Phase 5 R² Improvement Experiment")
    logger.info("=" * 60)

    # Load data
    logger.info("📊 Loading Phase 3 real data integration...")
    integrator = RealDataPhase3Integration()
    integrator.load_and_prepare_data()

    # Extract data
    x_spatial = integrator.spatiotemporal_embeddings
    x_genomic = integrator.genomic_embeddings
    x_temporal = integrator.temporal_embeddings
    y_motor = integrator.prognostic_targets[:, 0]
    y_cognitive = integrator.prognostic_targets[:, 1]

    logger.info(f"📈 Dataset: {x_spatial.shape[0]} patients")
    logger.info(f"🧬 Spatial: {x_spatial.shape[1:]} | Genomic: {x_genomic.shape[1:]}")
    logger.info(f"⏰ Temporal: {x_temporal.shape[1:]}")

    # Run improved training
    trainer = OptimizedTrainer(device="cpu")
    results = trainer.evaluate_loocv(
        x_spatial, x_genomic, x_temporal, y_motor, y_cognitive
    )

    # Display results
    logger.info("🎯 R² Improvement Results:")
    logger.info(f"   Motor R² = {results['motor_r2']:.4f}")
    logger.info(f"   Cognitive AUC = {results['cognitive_auc']:.4f}")

    # Compare with baseline
    baseline_r2 = -0.1157
    improvement = results["motor_r2"] - baseline_r2
    logger.info(f"🔧 R² Improvement: {improvement:+.4f}")

    if improvement > 0:
        logger.info("✅ ARCHITECTURE IMPROVEMENT SUCCESSFUL!")
    else:
        logger.info("⚠️ Further optimization needed")

    return results


if __name__ == "__main__":
    results = run_r2_improvement_experiment()

    print("\n🎉 R² Improvement Experiment Complete!")
    print("Original R²: -0.1157")
    print(f"Improved R²: {results['motor_r2']:.4f}")
    print(f"Improvement: {results['motor_r2'] - (-0.1157):+.4f}")
