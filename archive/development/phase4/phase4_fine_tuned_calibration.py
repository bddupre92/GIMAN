#!/usr/bin/env python3
"""GIMAN Phase 4: Fine-Tuned Regularization Calibration
===================================================

Based on initial calibration results showing that lower weight decay performs better,
implementing targeted fine-tuning around the most promising configurations.

Strategy:
- Experiment 1: Minimal Regularization (dropout 0.3, weight_decay 1e-5)
- Experiment 2: Light Dropout (dropout 0.4, weight_decay 1e-4)
- Experiment 3: Light Weight Decay (dropout 0.5, weight_decay 1e-5)
- Focus on configurations that showed best performance trends

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


class LightRegularizedGAT(nn.Module):
    """GAT with light regularization parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.3,
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

        # Linear transformation
        h = self.W(x)  # [batch_size, num_nodes, out_features]

        # Apply layer norm - works with any batch size
        h = self.layer_norm(h)

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

        # Aggregate features using attention weights
        h_out = torch.zeros_like(h)
        for head in range(self.num_heads):
            alpha_head = alpha[:, :, :, head]
            h_head = h[:, :, head, :]
            h_out[:, :, head, :] = torch.bmm(alpha_head, h_head)

        # Flatten heads dimension
        h_out = h_out.contiguous().view(batch_size, num_nodes, -1)

        return h_out


class LightRegularizedGIMAN(nn.Module):
    """GIMAN with light regularization parameters."""

    def __init__(
        self,
        spatial_dim: int = 256,
        genomic_dim: int = 256,
        temporal_dim: int = 256,
        embedding_dim: int = 32,
        gat_hidden_dim: int = 32,
        num_heads: int = 1,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Light regularization embedding layers
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

        # GAT with light dropout
        self.gat = LightRegularizedGAT(
            in_features=3 * embedding_dim,
            out_features=gat_hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
        )

        # Prediction heads with light regularization
        self.motor_head = nn.Sequential(
            nn.Linear(gat_hidden_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),  # No sigmoid for regression
        )

        self.cognitive_head = nn.Sequential(
            nn.Linear(gat_hidden_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Sigmoid for binary classification
        )

        # Initialize weights with standard variance
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with standard variance."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)  # Standard gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, spatial, genomic, temporal, adj_matrix):
        batch_size = spatial.size(0)

        # Get embeddings
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

        # Node-level predictions (architecturally sound approach)
        motor_pred = self.motor_head(gat_out).squeeze(-1)  # [batch_size, num_nodes]
        cognitive_pred = self.cognitive_head(gat_out).squeeze(
            -1
        )  # [batch_size, num_nodes]

        return motor_pred, cognitive_pred

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FineTunedCalibrator:
    """Fine-tuned regularization calibration with LOOCV evaluation."""

    def __init__(self, device="cpu"):
        self.device = device
        self.experiments = [
            {
                "name": "Minimal Regularization",
                "description": "dropout=0.3, weight_decay=1e-5",
                "dropout_rate": 0.3,
                "weight_decay": 1e-5,
            },
            {
                "name": "Light Dropout",
                "description": "dropout=0.4, weight_decay=1e-4",
                "dropout_rate": 0.4,
                "weight_decay": 1e-4,
            },
            {
                "name": "Light Weight Decay",
                "description": "dropout=0.5, weight_decay=1e-5",
                "dropout_rate": 0.5,
                "weight_decay": 1e-5,
            },
        ]

    def run_calibration(
        self, X_spatial, X_genomic, X_temporal, y_motor, y_cognitive, adj_matrix
    ):
        """Run fine-tuned regularization calibration."""
        results = {}

        logger.info("🎯 GIMAN FINE-TUNED REGULARIZATION CALIBRATION")
        logger.info("=" * 60)
        logger.info(f"📊 Dataset: {X_spatial.shape[0]} patients")
        logger.info("🔬 Evaluation: Leave-One-Out Cross-Validation")
        logger.info(f"🧪 Experiments: {len(self.experiments)}")
        logger.info("📋 Focus: Light regularization based on initial findings")
        logger.info("=" * 60)

        for i, experiment in enumerate(self.experiments, 1):
            logger.info(f"🧪 Experiment {i}/3: {experiment['name']}")
            logger.info(f"   Configuration: {experiment['description']}")

            # Run LOOCV for this configuration
            experiment_results = self._run_experiment(
                X_spatial,
                X_genomic,
                X_temporal,
                y_motor,
                y_cognitive,
                adj_matrix,
                experiment["dropout_rate"],
                experiment["weight_decay"],
            )

            results[experiment["name"]] = experiment_results

            logger.info(
                f"   Results: Motor R² = {experiment_results['motor_r2']:.4f}, "
                f"Cognitive AUC = {experiment_results['cognitive_auc']:.4f}"
            )
            logger.info("-" * 40)

        # Analyze and report best configuration
        self._analyze_results(results)

        return results

    def _run_experiment(
        self,
        X_spatial,
        X_genomic,
        X_temporal,
        y_motor,
        y_cognitive,
        adj_matrix,
        dropout_rate,
        weight_decay,
    ):
        """Run LOOCV for a specific regularization configuration."""
        n_samples = X_spatial.shape[0]
        loo = LeaveOneOut()

        motor_predictions = []
        motor_targets = []
        cognitive_predictions = []
        cognitive_targets = []

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_spatial)):
            if (fold + 1) % 20 == 0:  # Progress logging every 20 folds
                logger.info(f"   Fold {fold + 1}/{n_samples}")

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

            # Create model with specific configuration
            model = LightRegularizedGIMAN(dropout_rate=dropout_rate).to(self.device)

            # Train model
            self._train_fold(
                model,
                X_train_spatial,
                X_train_genomic,
                X_train_temporal,
                y_train_motor,
                y_train_cognitive,
                train_adj,
                weight_decay,
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

        return {
            "motor_r2": motor_r2,
            "cognitive_auc": cognitive_auc,
            "motor_predictions": motor_predictions,
            "motor_targets": motor_targets,
            "cognitive_predictions": cognitive_predictions,
            "cognitive_targets": cognitive_targets,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
        }

    def _train_fold(
        self,
        model,
        X_spatial,
        X_genomic,
        X_temporal,
        y_motor,
        y_cognitive,
        adj_matrix,
        weight_decay,
    ):
        """Train a single fold with specified weight decay."""
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10)

        best_loss = float("inf")
        patience_counter = 0
        max_patience = 20

        # Convert to tensors and add batch dimension
        spatial_tensor = torch.FloatTensor(X_spatial).unsqueeze(0).to(self.device)
        genomic_tensor = torch.FloatTensor(X_genomic).unsqueeze(0).to(self.device)
        temporal_tensor = torch.FloatTensor(X_temporal).unsqueeze(0).to(self.device)
        motor_tensor = torch.FloatTensor(y_motor).to(self.device)
        cognitive_tensor = torch.FloatTensor(y_cognitive).to(self.device)
        batch_adj = torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)

        model.train()
        for epoch in range(100):  # Max 100 epochs
            optimizer.zero_grad()

            motor_pred, cognitive_pred = model(
                spatial_tensor, genomic_tensor, temporal_tensor, batch_adj
            )

            # Node-level loss calculation
            motor_loss = F.mse_loss(motor_pred.squeeze(0), motor_tensor)
            cognitive_loss = F.binary_cross_entropy(
                cognitive_pred.squeeze(0), cognitive_tensor
            )
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
            spatial_tensor = torch.FloatTensor(X_spatial).unsqueeze(0).to(self.device)
            genomic_tensor = torch.FloatTensor(X_genomic).unsqueeze(0).to(self.device)
            temporal_tensor = torch.FloatTensor(X_temporal).unsqueeze(0).to(self.device)
            batch_adj = torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)

            motor_pred, cognitive_pred = model(
                spatial_tensor, genomic_tensor, temporal_tensor, batch_adj
            )

            # Extract single node prediction and ensure proper format
            motor_pred_scalar = motor_pred.squeeze().cpu().numpy()
            cognitive_pred_scalar = cognitive_pred.squeeze().cpu().numpy()

            return [
                motor_pred_scalar.item()
                if motor_pred_scalar.ndim == 0
                else motor_pred_scalar[0]
            ], [
                cognitive_pred_scalar.item()
                if cognitive_pred_scalar.ndim == 0
                else cognitive_pred_scalar[0]
            ]

    def _analyze_results(self, results):
        """Analyze and report the best regularization configuration."""
        logger.info("=" * 60)
        logger.info("📊 FINE-TUNED CALIBRATION RESULTS ANALYSIS")
        logger.info("=" * 60)

        # Find best configurations
        best_motor_config = max(results.items(), key=lambda x: x[1]["motor_r2"])
        best_cognitive_config = max(
            results.items(), key=lambda x: x[1]["cognitive_auc"]
        )
        best_combined_config = max(
            results.items(), key=lambda x: x[1]["motor_r2"] + x[1]["cognitive_auc"]
        )

        # Report individual results
        for name, result in results.items():
            logger.info(f"🧪 {name}:")
            logger.info(f"   Motor R²: {result['motor_r2']:.4f}")
            logger.info(f"   Cognitive AUC: {result['cognitive_auc']:.4f}")
            logger.info(
                f"   Combined Score: {result['motor_r2'] + result['cognitive_auc']:.4f}"
            )
            logger.info(
                f"   Config: dropout={result['dropout_rate']}, weight_decay={result['weight_decay']}"
            )

            # Performance assessment
            if result["motor_r2"] > 0.0 and result["cognitive_auc"] > 0.6:
                logger.info("   🎉 EXCELLENT: Both targets achieved!")
            elif result["motor_r2"] > 0.0:
                logger.info("   ✅ BREAKTHROUGH: Positive R² achieved!")
            elif result["motor_r2"] > -0.1:
                logger.info("   📈 PROMISING: Close to positive R²!")
            else:
                logger.info("   ⚠️  NEEDS IMPROVEMENT: Negative R²")
            logger.info("")

        # Report best configurations
        logger.info("🏆 BEST CONFIGURATIONS:")
        logger.info(
            f"Best Motor Performance: {best_motor_config[0]} (R² = {best_motor_config[1]['motor_r2']:.4f})"
        )
        logger.info(
            f"Best Cognitive Performance: {best_cognitive_config[0]} (AUC = {best_cognitive_config[1]['cognitive_auc']:.4f})"
        )
        logger.info(
            f"Best Combined Performance: {best_combined_config[0]} (Score = {best_combined_config[1]['motor_r2'] + best_combined_config[1]['cognitive_auc']:.4f})"
        )

        # Recommendations
        logger.info("")
        logger.info("📋 RECOMMENDATIONS:")
        if best_combined_config[1]["motor_r2"] > 0.0:
            logger.info("🎉 BREAKTHROUGH ACHIEVED! Positive R² found!")
            logger.info(f"   Optimal config: {best_combined_config[0]}")
            logger.info("   Ready for advanced model development and cohort expansion")
        elif best_combined_config[1]["motor_r2"] > -0.05:
            logger.info(
                "📈 VERY CLOSE! Consider minimal additional regularization reduction"
            )
            logger.info(f"   Best config so far: {best_combined_config[0]}")
            logger.info("   Recommend testing dropout=0.2 or weight_decay=1e-6")
        else:
            logger.info("⚠️  Consider alternative approaches:")
            logger.info("   - Even lighter regularization (dropout < 0.3)")
            logger.info("   - Larger embedding dimensions")
            logger.info("   - Different architecture modifications")

        logger.info("=" * 60)


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
    logger.info("🎬 GIMAN Phase 4: Fine-Tuned Regularization Calibration")
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

    # Run fine-tuned regularization calibration
    calibrator = FineTunedCalibrator(device=device)
    results = calibrator.run_calibration(
        X_spatial, X_genomic, X_temporal, y_motor, y_cognitive, adj_matrix
    )

    # Create sample model for parameter count
    sample_model = LightRegularizedGIMAN()
    logger.info(
        f"🏗️ Model architecture: ~{sample_model.count_parameters():,} parameters"
    )

    logger.info("🎯 Fine-tuned regularization calibration complete!")


if __name__ == "__main__":
    main()
