#!/usr/bin/env python3
"""GIMAN Phase 4: Ultra-Regularized System
=======================================

Building on stabilized system success, implementing maximum regularization
to achieve positive R² in LOOCV evaluation.

Strategy:
- Even smaller model (32-dim embeddings)
- Maximum dropout (0.7)
- Strong weight decay (1e-2)
- Batch normalization for stability
- Early stopping to prevent overfitting

Author: GIMAN Development Team
Date: September 2025
"""

import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

# Import our data integration system
import sys

sys.path.append(
    "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase3"
)
from phase3_1_real_data_integration import RealDataPhase3Integration

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UltraRegularizedGAT(nn.Module):
    """Ultra-lightweight GAT with maximum regularization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.7,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.dropout_rate = dropout

        # Single linear transformation with layer norm
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.layer_norm = nn.LayerNorm(out_features)

        # Minimal attention mechanism
        self.attention = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes = x.size(0), x.size(1)

        # Linear transformation with batch norm
        h = self.W(x)  # [batch_size, num_nodes, out_features]

        # Apply layer norm - works with any batch size
        h = self.layer_norm(
            h
        )  # LayerNorm works directly on [batch_size, num_nodes, out_features]

        h = h.view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Simplified attention computation
        h_i = h.unsqueeze(2)  # [batch_size, num_nodes, 1, num_heads, head_dim]
        h_j = h.unsqueeze(1)  # [batch_size, 1, num_nodes, num_heads, head_dim]

        # Concatenate for attention computation
        h_i_expanded = h_i.expand(-1, -1, num_nodes, -1, -1)
        h_j_expanded = h_j.expand(-1, num_nodes, -1, -1, -1)
        concat = torch.cat([h_i_expanded, h_j_expanded], dim=-1)

        # Apply attention mechanism
        e = self.attention(concat).squeeze(
            -1
        )  # [batch_size, num_nodes, num_nodes, num_heads]
        e = self.leaky_relu(e)

        # Apply adjacency matrix mask
        adj_expanded = adj_matrix.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        e = e.masked_fill(adj_expanded == 0, -1e9)

        # Softmax attention weights
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        # Aggregate features using attention weights - simpler approach
        # alpha: [batch_size, num_nodes, num_nodes, num_heads]
        # h: [batch_size, num_nodes, num_heads, head_dim]
        h_out = torch.zeros_like(h)
        for head in range(self.num_heads):
            # Get attention weights for this head: [batch_size, num_nodes, num_nodes]
            alpha_head = alpha[:, :, :, head]
            # Get features for this head: [batch_size, num_nodes, head_dim]
            h_head = h[:, :, head, :]
            # Apply attention: [batch_size, num_nodes, num_nodes] @ [batch_size, num_nodes, head_dim]
            h_out[:, :, head, :] = torch.bmm(alpha_head, h_head)

        # Flatten heads dimension
        h_out = h_out.contiguous().view(batch_size, num_nodes, -1)

        return h_out


class UltraRegularizedGIMAN(nn.Module):
    """Ultra-regularized GIMAN with minimal parameters."""

    def __init__(
        self,
        spatial_dim: int = 256,
        genomic_dim: int = 256,
        temporal_dim: int = 256,
        embedding_dim: int = 32,  # Even smaller!
        gat_hidden_dim: int = 32,
        num_heads: int = 1,  # Single head
        dropout_rate: float = 0.7,
    ):  # Maximum dropout
        super().__init__()

        self.embedding_dim = embedding_dim

        # Ultra-lightweight embedding layers with layer norm
        self.spatial_embedding = nn.Sequential(
            nn.Linear(spatial_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.genomic_embedding = nn.Sequential(
            nn.Linear(genomic_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.temporal_embedding = nn.Sequential(
            nn.Linear(temporal_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Minimal GAT
        self.gat = UltraRegularizedGAT(
            in_features=3 * embedding_dim,
            out_features=gat_hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
        )

        # Minimal prediction heads
        self.motor_head = nn.Sequential(
            nn.Linear(gat_hidden_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
        )

        self.cognitive_head = nn.Sequential(
            nn.Linear(gat_hidden_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with smaller variance."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)  # Smaller gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, spatial, genomic, temporal, adj_matrix):
        batch_size = spatial.size(0)

        # Get embeddings with batch norm
        spatial_emb = self.spatial_embedding(spatial.view(-1, spatial.size(-1)))
        spatial_emb = spatial_emb.view(batch_size, -1, self.embedding_dim)

        genomic_emb = self.genomic_embedding(genomic.view(-1, genomic.size(-1)))
        genomic_emb = genomic_emb.view(batch_size, -1, self.embedding_dim)

        temporal_emb = self.temporal_embedding(temporal.view(-1, temporal.size(-1)))
        temporal_emb = temporal_emb.view(batch_size, -1, self.embedding_dim)

        # Concatenate modalities
        node_features = torch.cat([spatial_emb, genomic_emb, temporal_emb], dim=-1)

        # Graph attention
        gat_out = self.gat(node_features, adj_matrix)

        # Node-level predictions (no global pooling)
        motor_pred = self.motor_head(gat_out).squeeze(-1)  # [batch_size, num_nodes]
        cognitive_pred = self.cognitive_head(gat_out).squeeze(
            -1
        )  # [batch_size, num_nodes]

        return motor_pred, cognitive_pred

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LOOCVEvaluator:
    """Leave-One-Out Cross-Validation evaluator with early stopping."""

    def __init__(self, device="cpu"):
        self.device = device
        self.scaler = StandardScaler()

    def evaluate(
        self, X_spatial, X_genomic, X_temporal, y_motor, y_cognitive, adj_matrix
    ):
        """Run LOOCV evaluation."""
        n_samples = X_spatial.shape[0]
        loo = LeaveOneOut()

        motor_predictions = []
        motor_targets = []
        cognitive_predictions = []
        cognitive_targets = []

        logger.info(f"Starting LOOCV evaluation with {n_samples} patients...")

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_spatial)):
            logger.info(f"Fold {fold + 1}/{n_samples}")

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
            train_adj, test_adj = (
                adj_matrix[train_idx][:, train_idx],
                adj_matrix[test_idx][:, test_idx],
            )

            # Create model
            model = UltraRegularizedGIMAN().to(self.device)
            logger.info(
                f"Ultra-regularized GIMAN initialized with {model.count_parameters():,} parameters"
            )

            # Train model with early stopping
            self._train_fold(
                model,
                X_train_spatial,
                X_train_genomic,
                X_train_temporal,
                y_train_motor,
                y_train_cognitive,
                train_adj,
            )

            # Test model
            motor_pred, cognitive_pred = self._test_fold(
                model, X_test_spatial, X_test_genomic, X_test_temporal, test_adj
            )

            motor_predictions.extend(motor_pred)
            motor_targets.extend(y_test_motor)
            cognitive_predictions.extend(cognitive_pred)
            cognitive_targets.extend(y_test_cognitive)

        # Calculate metrics
        motor_r2 = r2_score(motor_targets, motor_predictions)
        cognitive_auc = roc_auc_score(cognitive_targets, cognitive_predictions)

        logger.info(
            f"LOOCV Results: Motor R² = {motor_r2:.4f}, Cognitive AUC = {cognitive_auc:.4f}"
        )

        return {
            "motor_r2": motor_r2,
            "cognitive_auc": cognitive_auc,
            "motor_predictions": motor_predictions,
            "motor_targets": motor_targets,
            "cognitive_predictions": cognitive_predictions,
            "cognitive_targets": cognitive_targets,
        }

    def _train_fold(
        self, model, X_spatial, X_genomic, X_temporal, y_motor, y_cognitive, adj_matrix
    ):
        """Train a single fold with early stopping."""
        optimizer = Adam(
            model.parameters(), lr=0.001, weight_decay=1e-2
        )  # Strong weight decay
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10)

        best_loss = float("inf")
        patience_counter = 0
        max_patience = 20

        # Convert to tensors and add batch dimension for graph structure
        spatial_tensor = (
            torch.FloatTensor(X_spatial).unsqueeze(0).to(self.device)
        )  # [1, num_nodes, features]
        genomic_tensor = (
            torch.FloatTensor(X_genomic).unsqueeze(0).to(self.device)
        )  # [1, num_nodes, features]
        temporal_tensor = (
            torch.FloatTensor(X_temporal).unsqueeze(0).to(self.device)
        )  # [1, num_nodes, features]
        motor_tensor = torch.FloatTensor(y_motor).to(
            self.device
        )  # Keep as [num_nodes] for loss calculation
        cognitive_tensor = torch.FloatTensor(y_cognitive).to(
            self.device
        )  # Keep as [num_nodes] for loss calculation
        batch_adj = (
            torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)
        )  # [1, num_nodes, num_nodes]

        model.train()
        for epoch in range(100):  # Max 100 epochs
            optimizer.zero_grad()

            motor_pred, cognitive_pred = model(
                spatial_tensor, genomic_tensor, temporal_tensor, batch_adj
            )

            # Node-level loss calculation (no target aggregation)
            motor_loss = F.mse_loss(
                motor_pred.squeeze(0), motor_tensor
            )  # Direct node-to-node comparison
            cognitive_loss = F.binary_cross_entropy(
                cognitive_pred.squeeze(0), cognitive_tensor
            )  # Direct node-to-node comparison
            total_loss = 0.5 * motor_loss + 0.5 * cognitive_loss

            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss)

            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break

    def _test_fold(self, model, X_spatial, X_genomic, X_temporal, adj_matrix):
        """Test a single fold."""
        model.eval()

        with torch.no_grad():
            spatial_tensor = (
                torch.FloatTensor(X_spatial).unsqueeze(0).to(self.device)
            )  # [1, 1, features]
            genomic_tensor = (
                torch.FloatTensor(X_genomic).unsqueeze(0).to(self.device)
            )  # [1, 1, features]
            temporal_tensor = (
                torch.FloatTensor(X_temporal).unsqueeze(0).to(self.device)
            )  # [1, 1, features]
            batch_adj = (
                torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)
            )  # [1, 1, 1]

            motor_pred, cognitive_pred = model(
                spatial_tensor, genomic_tensor, temporal_tensor, batch_adj
            )

            # Extract single node prediction (test case has only 1 node) and ensure it's a scalar
            motor_pred_scalar = motor_pred.squeeze().cpu().numpy()
            cognitive_pred_scalar = cognitive_pred.squeeze().cpu().numpy()

            # Convert to list if scalar to ensure proper iteration
            return [
                motor_pred_scalar.item()
                if motor_pred_scalar.ndim == 0
                else motor_pred_scalar[0]
            ], [
                cognitive_pred_scalar.item()
                if cognitive_pred_scalar.ndim == 0
                else cognitive_pred_scalar[0]
            ]


def validate_data_quality(spatial, genomic, temporal, y_motor, y_cognitive):
    """Validate data quality."""
    logger.info("🔍 Validating data quality...")

    # Check for NaN values
    spatial_nan = np.isnan(spatial).sum()
    genomic_nan = np.isnan(genomic).sum()
    temporal_nan = np.isnan(temporal).sum()
    motor_nan = np.isnan(y_motor).sum()
    cognitive_nan = np.isnan(y_cognitive).sum()

    logger.info(
        f"NaN counts - Spatial: {spatial_nan}, Genomic: {genomic_nan}, Temporal: {temporal_nan}, Motor: {motor_nan}, Cognitive: {cognitive_nan}"
    )

    # Check for infinite values
    spatial_inf = np.isinf(spatial).sum()
    genomic_inf = np.isinf(genomic).sum()
    temporal_inf = np.isinf(temporal).sum()

    logger.info(
        f"Inf counts - Spatial: {spatial_inf}, Genomic: {genomic_inf}, Temporal: {temporal_inf}"
    )

    # Check data ranges
    logger.info("Data ranges:")
    logger.info(f"  Spatial: [{spatial.min():.3f}, {spatial.max():.3f}]")
    logger.info(f"  Genomic: [{genomic.min():.3f}, {genomic.max():.3f}]")
    logger.info(f"  Temporal: [{temporal.min():.3f}, {temporal.max():.3f}]")
    logger.info(f"  Motor: [{y_motor.min():.3f}, {y_motor.max():.3f}]")
    logger.info(f"  Cognitive: [{y_cognitive.min():.3f}, {y_cognitive.max():.3f}]")

    # Check cognitive conversion rate
    conversion_rate = np.mean(y_cognitive) * 100
    logger.info(f"Cognitive conversion rate: {conversion_rate:.1f}%")

    # Validation check
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
        logger.warning(f"❌ Data validation FAILED - {total_issues} issues detected")
        return False


def main():
    """Main execution function."""
    logger.info("🎬 GIMAN Phase 4: Ultra-Regularized System")
    logger.info("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("📊 Loading and preparing data...")
    integrator = RealDataPhase3Integration()
    integrator.load_and_prepare_data()

    # Extract data
    X_spatial = integrator.spatiotemporal_embeddings
    X_genomic = integrator.genomic_embeddings
    X_temporal = integrator.temporal_embeddings
    y_motor = integrator.prognostic_targets[:, 0]  # Motor progression
    y_cognitive = integrator.prognostic_targets[:, 1]  # Cognitive conversion
    adj_matrix = integrator.similarity_matrix

    # Validate data quality
    if not validate_data_quality(
        X_spatial, X_genomic, X_temporal, y_motor, y_cognitive
    ):
        logger.error("Data validation failed. Stopping execution.")
        return

    logger.info(f"Dataset prepared: {X_spatial.shape[0]} patients")

    # Run LOOCV evaluation
    logger.info("🔄 Starting Leave-One-Out Cross-Validation...")
    evaluator = LOOCVEvaluator(device=device)
    results = evaluator.evaluate(
        X_spatial, X_genomic, X_temporal, y_motor, y_cognitive, adj_matrix
    )

    # Display results
    logger.info("=" * 60)
    logger.info("🎯 ULTRA-REGULARIZED GIMAN RESULTS")
    logger.info("=" * 60)
    logger.info(f"📊 Dataset: {X_spatial.shape[0]} patients")

    # Create a sample model to count parameters
    sample_model = UltraRegularizedGIMAN()
    logger.info(
        f"🏗️ Model: Ultra-regularized architecture (~{sample_model.count_parameters():,} parameters)"
    )
    logger.info("🔬 Evaluation: Leave-One-Out Cross-Validation")
    logger.info(f"📈 Motor progression R²: {results['motor_r2']:.4f}")
    logger.info(f"🧠 Cognitive conversion AUC: {results['cognitive_auc']:.4f}")

    if results["motor_r2"] > 0.0 and results["cognitive_auc"] > 0.6:
        logger.info("🎉 SUCCESS: Target metrics achieved!")
        logger.info("📋 Ready for advanced model development")
    elif results["motor_r2"] > 0.0:
        logger.info("✅ PROGRESS: Positive R² achieved!")
        logger.info("📋 Motor prediction stable, cognitive needs improvement")
    else:
        logger.info("❌ FURTHER REGULARIZATION NEEDED")
        logger.info("📋 Consider ensemble methods or data expansion")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
