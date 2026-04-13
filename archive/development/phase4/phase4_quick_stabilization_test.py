#!/usr/bin/env python3
"""GIMAN Phase 4: Quick Stabilization Test

Quick test to validate our fixes:
1. Temporal embedding NaN fix
2. Simplified model architecture
3. Basic train/test split validation

This will help us verify our approach before implementing full LOOCV.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Import data integration
archive_phase3_path = Path(__file__).parent.parent / "phase3"
sys.path.append(str(archive_phase3_path))
from phase3_1_real_data_integration import RealDataPhase3Integration

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleGIMANBaseline(nn.Module):
    """Very simple GIMAN baseline to test our fixes."""

    def __init__(self, embed_dim: int = 32, dropout: float = 0.5):
        super().__init__()

        # Simple projections
        self.spatial_proj = nn.Linear(256, embed_dim)
        self.genomic_proj = nn.Linear(256, embed_dim)
        self.temporal_proj = nn.Linear(256, embed_dim)

        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # Simple predictors
        self.motor_pred = nn.Linear(embed_dim, 1)
        self.cognitive_pred = nn.Linear(embed_dim, 1)

        self.dropout = nn.Dropout(dropout)

        logger.info(f"Simple GIMAN baseline: {self.count_parameters():,} parameters")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, spatial, genomic, temporal):
        # Project embeddings
        spatial_emb = self.dropout(F.relu(self.spatial_proj(spatial)))
        genomic_emb = self.dropout(F.relu(self.genomic_proj(genomic)))
        temporal_emb = self.dropout(F.relu(self.temporal_proj(temporal)))

        # Concatenate and fuse
        combined = torch.cat([spatial_emb, genomic_emb, temporal_emb], dim=-1)
        fused = self.fusion(combined)

        # Predictions
        motor_out = self.motor_pred(fused).squeeze(-1)
        cognitive_out = self.cognitive_pred(fused).squeeze(-1)

        return motor_out, cognitive_out


def validate_data_quality(
    spatial_emb, genomic_emb, temporal_emb, motor_targets, cognitive_targets
):
    """Validate data quality and report issues."""
    logger.info("🔍 Validating data quality...")

    # Check for NaN values
    spatial_nan = np.isnan(spatial_emb).sum()
    genomic_nan = np.isnan(genomic_emb).sum()
    temporal_nan = np.isnan(temporal_emb).sum()
    motor_nan = np.isnan(motor_targets).sum()
    cognitive_nan = np.isnan(cognitive_targets).sum()

    logger.info(
        f"NaN counts - Spatial: {spatial_nan}, Genomic: {genomic_nan}, "
        f"Temporal: {temporal_nan}, Motor: {motor_nan}, Cognitive: {cognitive_nan}"
    )

    # Check for infinite values
    spatial_inf = np.isinf(spatial_emb).sum()
    genomic_inf = np.isinf(genomic_emb).sum()
    temporal_inf = np.isinf(temporal_emb).sum()

    logger.info(
        f"Inf counts - Spatial: {spatial_inf}, Genomic: {genomic_inf}, Temporal: {temporal_inf}"
    )

    # Check data ranges
    logger.info("Data ranges:")
    logger.info(f"  Spatial: [{spatial_emb.min():.3f}, {spatial_emb.max():.3f}]")
    logger.info(f"  Genomic: [{genomic_emb.min():.3f}, {genomic_emb.max():.3f}]")
    logger.info(f"  Temporal: [{temporal_emb.min():.3f}, {temporal_emb.max():.3f}]")
    logger.info(f"  Motor: [{motor_targets.min():.3f}, {motor_targets.max():.3f}]")
    logger.info(
        f"  Cognitive: [{cognitive_targets.min():.3f}, {cognitive_targets.max():.3f}]"
    )

    # Check cognitive class balance
    positive_rate = cognitive_targets.mean()
    logger.info(f"Cognitive conversion rate: {positive_rate:.1%}")

    # Validation status
    total_issues = (
        spatial_nan
        + genomic_nan
        + temporal_nan
        + motor_nan
        + cognitive_nan
        + spatial_inf
        + genomic_inf
        + temporal_inf
    )

    if total_issues == 0:
        logger.info("✅ Data validation PASSED - no issues detected")
        return True
    else:
        logger.warning(f"⚠️ Data validation found {total_issues} issues")
        return False


def main():
    """Quick stabilization test."""
    logger.info("🎬 GIMAN Phase 4: Quick Stabilization Test")
    logger.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and prepare data
    logger.info("📊 Loading and preparing data...")
    data_integrator = RealDataPhase3Integration()
    data_integrator.load_and_prepare_data()

    # Validate data quality
    data_valid = validate_data_quality(
        data_integrator.spatiotemporal_embeddings,
        data_integrator.genomic_embeddings,
        data_integrator.temporal_embeddings,
        data_integrator.prognostic_targets[:, 0],  # motor
        data_integrator.prognostic_targets[:, 1],  # cognitive
    )

    if not data_valid:
        logger.error("❌ Data validation failed - cannot proceed with training")
        return None

    # Prepare data
    spatial_emb = data_integrator.spatiotemporal_embeddings
    genomic_emb = data_integrator.genomic_embeddings
    temporal_emb = data_integrator.temporal_embeddings
    motor_targets = data_integrator.prognostic_targets[:, 0]
    cognitive_targets = data_integrator.prognostic_targets[:, 1]

    # Standardize features
    scaler_spatial = StandardScaler()
    scaler_genomic = StandardScaler()
    scaler_temporal = StandardScaler()
    scaler_motor = StandardScaler()

    spatial_scaled = scaler_spatial.fit_transform(spatial_emb)
    genomic_scaled = scaler_genomic.fit_transform(genomic_emb)
    temporal_scaled = scaler_temporal.fit_transform(temporal_emb)
    motor_scaled = scaler_motor.fit_transform(motor_targets.reshape(-1, 1)).flatten()

    logger.info(f"Dataset prepared: {len(spatial_scaled)} patients")

    # Train/test split
    indices = np.arange(len(spatial_scaled))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=cognitive_targets
    )

    X_train_spatial = spatial_scaled[train_idx]
    X_train_genomic = genomic_scaled[train_idx]
    X_train_temporal = temporal_scaled[train_idx]
    y_train_motor = motor_scaled[train_idx]
    y_train_cognitive = cognitive_targets[train_idx]

    X_test_spatial = spatial_scaled[test_idx]
    X_test_genomic = genomic_scaled[test_idx]
    X_test_temporal = temporal_scaled[test_idx]
    y_test_motor = motor_scaled[test_idx]
    y_test_cognitive = cognitive_targets[test_idx]

    logger.info(f"Train set: {len(train_idx)} patients")
    logger.info(f"Test set: {len(test_idx)} patients")

    # Convert to tensors
    X_train_spatial_t = torch.tensor(X_train_spatial, dtype=torch.float32).to(device)
    X_train_genomic_t = torch.tensor(X_train_genomic, dtype=torch.float32).to(device)
    X_train_temporal_t = torch.tensor(X_train_temporal, dtype=torch.float32).to(device)
    y_train_motor_t = torch.tensor(y_train_motor, dtype=torch.float32).to(device)
    y_train_cognitive_t = torch.tensor(y_train_cognitive, dtype=torch.float32).to(
        device
    )

    X_test_spatial_t = torch.tensor(X_test_spatial, dtype=torch.float32).to(device)
    X_test_genomic_t = torch.tensor(X_test_genomic, dtype=torch.float32).to(device)
    X_test_temporal_t = torch.tensor(X_test_temporal, dtype=torch.float32).to(device)
    y_test_motor_t = torch.tensor(y_test_motor, dtype=torch.float32).to(device)
    y_test_cognitive_t = torch.tensor(y_test_cognitive, dtype=torch.float32).to(device)

    # Initialize model
    model = SimpleGIMANBaseline(embed_dim=32, dropout=0.5).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    motor_criterion = nn.MSELoss()
    cognitive_criterion = nn.BCEWithLogitsLoss()

    # Training loop
    logger.info("🚀 Starting training...")
    model.train()

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(100):
        optimizer.zero_grad()

        motor_pred, cognitive_pred = model(
            X_train_spatial_t, X_train_genomic_t, X_train_temporal_t
        )

        motor_loss = motor_criterion(motor_pred, y_train_motor_t)
        cognitive_loss = cognitive_criterion(cognitive_pred, y_train_cognitive_t)

        total_loss = motor_loss + cognitive_loss
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_motor_pred, val_cognitive_pred = model(
                    X_test_spatial_t, X_test_genomic_t, X_test_temporal_t
                )
                val_motor_loss = motor_criterion(val_motor_pred, y_test_motor_t)
                val_cognitive_loss = cognitive_criterion(
                    val_cognitive_pred, y_test_cognitive_t
                )
                val_total_loss = val_motor_loss + val_cognitive_loss

                logger.info(
                    f"Epoch {epoch}: Train Loss = {total_loss.item():.4f}, Val Loss = {val_total_loss.item():.4f}"
                )

                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            model.train()

    # Final evaluation
    logger.info("🧪 Final evaluation...")
    model.eval()

    with torch.no_grad():
        test_motor_pred, test_cognitive_pred = model(
            X_test_spatial_t, X_test_genomic_t, X_test_temporal_t
        )

        # Convert to numpy
        test_motor_pred_np = test_motor_pred.cpu().numpy()
        test_cognitive_pred_np = torch.sigmoid(test_cognitive_pred).cpu().numpy()

        # Calculate metrics
        motor_r2 = r2_score(y_test_motor, test_motor_pred_np)

        try:
            cognitive_auc = roc_auc_score(y_test_cognitive, test_cognitive_pred_np)
        except ValueError:
            cognitive_auc = 0.5

        # Results
        logger.info("=" * 50)
        logger.info("🎯 QUICK STABILIZATION TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"📊 Dataset: {len(spatial_scaled)} patients")
        logger.info(
            f"🏗️ Model: Simple baseline (~{model.count_parameters():,} parameters)"
        )
        logger.info(f"📈 Motor progression R²: {motor_r2:.4f}")
        logger.info(f"🧠 Cognitive conversion AUC: {cognitive_auc:.4f}")

        # Status assessment
        if motor_r2 > 0.0 and cognitive_auc > 0.6:
            logger.info("✅ SUCCESS: Both metrics show improvement!")
            logger.info("📋 Next step: Implement full LOOCV with this architecture")
        elif motor_r2 > 0.0:
            logger.info("✅ PARTIAL SUCCESS: Positive R² achieved!")
            logger.info("📋 Motor prediction working, cognitive needs tuning")
        else:
            logger.info("⚠️ STILL OVERFITTING: Need further simplification")
            logger.info("📋 Consider even simpler model or more regularization")

        logger.info("=" * 50)

        results = {
            "motor_r2": motor_r2,
            "cognitive_auc": cognitive_auc,
            "data_clean": data_valid,
            "model_parameters": model.count_parameters(),
        }

        return results


if __name__ == "__main__":
    main()
