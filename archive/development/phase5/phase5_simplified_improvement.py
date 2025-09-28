#!/usr/bin/env python3
"""Phase 5 REAL R² Improvement: Simplified Approach.

The previous "improvement" made R² worse (-0.4132 vs -0.1157).
This happens because:
1. Small dataset (95 patients) + complex architecture = overfitting
2. BatchNorm with single test samples causes instability
3. Too many parameters for limited data

REAL improvement strategy:
1. SIMPLIFY the architecture (fewer parameters)
2. Remove BatchNorm (problematic with single samples)
3. Focus on regularization over complexity
4. Ensemble methods for better generalization
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


class SimplifiedGIMAN(nn.Module):
    """SIMPLIFIED GIMAN - Fewer parameters, better generalization."""

    def __init__(
        self,
        spatial_dim: int = 256,
        genomic_dim: int = 256,
        temporal_dim: int = 256,
        embed_dim: int = 32,  # REDUCED from 64
        num_heads: int = 2,  # REDUCED from 4
        dropout: float = 0.2,  # REDUCED from 0.3
    ):
        """Initialize simplified GIMAN."""
        super().__init__()
        self.embed_dim = embed_dim

        # SIMPLE embedding layers (NO BatchNorm)
        self.spatial_embedding = nn.Sequential(
            nn.Linear(spatial_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.genomic_embedding = nn.Sequential(
            nn.Linear(genomic_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal_embedding = nn.Sequential(
            nn.Linear(temporal_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Simple attention (NO complex multihead)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # SIMPLE task towers (NO deep stacking)
        self.motor_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),  # Motor regression
        )

        self.cognitive_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),  # Cognitive classification
        )

        # Simple weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Simple Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, spatial, genomic, temporal, adj_matrix=None):
        """Simple forward pass."""
        # Simple embeddings
        spatial_emb = self.spatial_embedding(spatial)
        genomic_emb = self.genomic_embedding(genomic)
        temporal_emb = self.temporal_embedding(temporal)

        # Simple concatenation + attention
        combined = torch.cat([spatial_emb, genomic_emb, temporal_emb], dim=-1)
        attended = self.attention(combined)

        # Task predictions
        motor_pred = self.motor_tower(attended)
        cognitive_pred = self.cognitive_tower(attended)

        return motor_pred, cognitive_pred


class EnsembleGIMAN:
    """Ensemble of simplified models for better generalization."""

    def __init__(self, n_models=3, device="cpu"):
        self.n_models = n_models
        self.device = device
        self.models = []

    def train_ensemble(self, x_spatial, x_genomic, x_temporal, y_motor, y_cognitive):
        """Train ensemble of simplified models."""
        self.models = []

        for i in range(self.n_models):
            logger.info(f"🔧 Training ensemble model {i + 1}/{self.n_models}")

            # Create model with slight variations
            model = SimplifiedGIMAN(
                spatial_dim=x_spatial.shape[1],
                genomic_dim=x_genomic.shape[1],
                temporal_dim=x_temporal.shape[1],
                embed_dim=32 + i * 4,  # Slight variation
                dropout=0.15 + i * 0.05,  # Slight variation
            ).to(self.device)

            # Train with simple settings
            model = self._train_single_model(
                model, x_spatial, x_genomic, x_temporal, y_motor, y_cognitive
            )
            self.models.append(model)

        return self

    def _train_single_model(
        self, model, x_spatial, x_genomic, x_temporal, y_motor, y_cognitive
    ):
        """Train single model with simple hyperparameters."""
        # Convert to tensors
        x_spatial_tensor = torch.FloatTensor(x_spatial).to(self.device)
        x_genomic_tensor = torch.FloatTensor(x_genomic).to(self.device)
        x_temporal_tensor = torch.FloatTensor(x_temporal).to(self.device)
        y_motor_tensor = torch.FloatTensor(y_motor).to(self.device)
        y_cognitive_tensor = torch.FloatTensor(y_cognitive).to(self.device)

        # Simple optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

        # Loss functions
        motor_criterion = nn.MSELoss()
        cognitive_criterion = nn.BCELoss()

        # Simple training loop
        model.train()
        for epoch in range(100):  # Fewer epochs
            optimizer.zero_grad()

            # Forward pass
            motor_pred, cognitive_pred = model(
                x_spatial_tensor, x_genomic_tensor, x_temporal_tensor
            )

            # Simple loss
            motor_loss = motor_criterion(motor_pred.squeeze(), y_motor_tensor)
            cognitive_loss = cognitive_criterion(
                cognitive_pred.squeeze(), y_cognitive_tensor
            )
            total_loss = 0.7 * motor_loss + 0.3 * cognitive_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

        return model

    def predict(self, x_spatial, x_genomic, x_temporal):
        """Ensemble prediction."""
        motor_preds = []
        cognitive_preds = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                x_spatial_tensor = torch.FloatTensor(x_spatial).to(self.device)
                x_genomic_tensor = torch.FloatTensor(x_genomic).to(self.device)
                x_temporal_tensor = torch.FloatTensor(x_temporal).to(self.device)

                motor_pred, cognitive_pred = model(
                    x_spatial_tensor, x_genomic_tensor, x_temporal_tensor
                )

                motor_preds.append(motor_pred.cpu().numpy())
                cognitive_preds.append(cognitive_pred.cpu().numpy())

        # Average predictions
        motor_pred_avg = np.mean(motor_preds, axis=0)
        cognitive_pred_avg = np.mean(cognitive_preds, axis=0)

        return motor_pred_avg, cognitive_pred_avg


def run_simplified_r2_improvement():
    """Run simplified R² improvement experiment."""
    logger.info("🚀 Starting SIMPLIFIED R² Improvement Experiment")
    logger.info("🎯 Strategy: SIMPLIFY architecture for small dataset")
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

    # Run simplified LOOCV
    loo = LeaveOneOut()
    n_samples = x_spatial.shape[0]

    motor_predictions = []
    motor_actuals = []
    cognitive_predictions = []
    cognitive_actuals = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(x_spatial)):
        if fold % 10 == 0:
            logger.info(f"📊 Processing fold {fold + 1}/{n_samples}")

        # Split data
        X_train_spatial, X_test_spatial = x_spatial[train_idx], x_spatial[test_idx]
        X_train_genomic, X_test_genomic = x_genomic[train_idx], x_genomic[test_idx]
        X_train_temporal, X_test_temporal = x_temporal[train_idx], x_temporal[test_idx]
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

        # Train ensemble
        ensemble = EnsembleGIMAN(n_models=3, device="cpu")
        ensemble.train_ensemble(
            X_train_spatial,
            X_train_genomic,
            X_train_temporal,
            y_train_motor,
            y_train_cognitive,
        )

        # Predict
        motor_pred, cognitive_pred = ensemble.predict(
            X_test_spatial, X_test_genomic, X_test_temporal
        )

        motor_predictions.append(motor_pred)
        motor_actuals.append(y_test_motor)
        cognitive_predictions.append(cognitive_pred)
        cognitive_actuals.append(y_test_cognitive)

    # Calculate metrics
    motor_pred_array = np.array(motor_predictions).flatten()
    motor_actual_array = np.array(motor_actuals).flatten()
    cognitive_pred_array = np.array(cognitive_predictions).flatten()
    cognitive_actual_array = np.array(cognitive_actuals).flatten()

    # Final metrics
    motor_r2 = r2_score(motor_actual_array, motor_pred_array)
    try:
        cognitive_auc = roc_auc_score(cognitive_actual_array, cognitive_pred_array)
    except ValueError:
        cognitive_auc = 0.5

    logger.info("✅ SIMPLIFIED R² Improvement Complete!")
    logger.info(f"🎯 Simplified Motor R² = {motor_r2:.4f}")
    logger.info(f"🧠 Simplified Cognitive AUC = {cognitive_auc:.4f}")

    # Compare with original
    original_r2 = -0.1157
    complex_r2 = -0.4132
    improvement_vs_original = motor_r2 - original_r2
    improvement_vs_complex = motor_r2 - complex_r2

    logger.info(f"🔧 vs Original: {improvement_vs_original:+.4f}")
    logger.info(f"🔧 vs Complex:  {improvement_vs_complex:+.4f}")

    if motor_r2 > original_r2:
        logger.info("✅ SIMPLIFIED APPROACH SUCCESSFUL!")
    else:
        logger.info("⚠️ Still needs optimization")

    return {
        "motor_r2": motor_r2,
        "cognitive_auc": cognitive_auc,
        "improvement_vs_original": improvement_vs_original,
        "improvement_vs_complex": improvement_vs_complex,
    }


if __name__ == "__main__":
    results = run_simplified_r2_improvement()

    print("\n🎉 SIMPLIFIED R² Improvement Complete!")
    print("Original R²: -0.1157")
    print("Complex R²:  -0.4132 (WORSE)")
    print(f"Simplified:  {results['motor_r2']:.4f}")
    print(f"Improvement: {results['improvement_vs_original']:+.4f}")
