#!/usr/bin/env python3
"""Phase 4 Enhanced Unified GIMAN System
=====================================

Enhanced version with advanced training strategies, better architecture,
and optimization improvements for the unified multimodal system.

Key improvements:
- Multi-head self-attention with residual connections
- Advanced training with learning rate scheduling and early stopping
- Enhanced ensemble predictors with batch normalization
- Improved regularization and gradient handling
- Better interpretability mechanisms

Author: GIMAN Development Team
Date: September 24, 2025
"""

import logging
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Optional interpretability imports
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Import our previous phase models
sys.path.append(".")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedConfig:
    """Enhanced configuration for Phase 4 system."""

    # Model architecture
    embed_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.3

    # Training parameters
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0

    # Advanced training
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    early_stopping_patience: int = 25
    warmup_epochs: int = 5

    # Batch and regularization
    batch_size: int = 16
    label_smoothing: float = 0.1

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EnhancedUnifiedAttentionModule(nn.Module):
    """Enhanced unified attention with self-attention and residual connections."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Feature processing with residual connections
        self.feature_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Multi-head self-attention for feature interaction
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Cross-modal attention with learnable temperature
        self.attention_weights = nn.Parameter(torch.randn(3, embed_dim))
        self.attention_temperature = nn.Parameter(torch.ones(1))

        # Enhanced importance weighting
        self.importance_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid(),
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, spatial_features, genomic_features, temporal_features):
        """Enhanced forward pass with attention mechanisms and residual connections."""
        batch_size = spatial_features.size(0)

        # Process each modality with residual connections
        spatial_processed = self.feature_processor(spatial_features) + spatial_features
        genomic_processed = self.feature_processor(genomic_features) + genomic_features
        temporal_processed = (
            self.feature_processor(temporal_features) + temporal_features
        )

        # Stack for attention computation
        stacked_features = torch.stack(
            [spatial_processed, genomic_processed, temporal_processed], dim=1
        )  # [batch_size, 3, embed_dim]

        # Apply self-attention for feature interaction
        attended_features, attention_weights = self.self_attention(
            stacked_features, stacked_features, stacked_features
        )

        # Residual connection with self-attention
        attended_features = attended_features + stacked_features
        attended_features = self.layer_norm(attended_features)

        # Compute cross-modal attention weights with temperature scaling
        attention_scores = (
            torch.einsum("bmd,md->bm", attended_features, self.attention_weights)
            / self.attention_temperature
        )
        cross_modal_weights = F.softmax(attention_scores, dim=1)

        # Apply attention
        unified_features = torch.einsum(
            "bm,bmd->bd", cross_modal_weights, attended_features
        )

        # Importance weighting
        importance_weights = self.importance_net(unified_features)
        weighted_features = unified_features * importance_weights

        return weighted_features, cross_modal_weights, attention_weights


class EnhancedEnsemblePredictor(nn.Module):
    """Enhanced ensemble predictor with batch normalization and better regularization."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        # Enhanced motor prediction head with batch normalization
        self.motor_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Enhanced cognitive prediction head with batch normalization
        self.cognitive_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        """Forward pass through ensemble predictors."""
        motor_pred = self.motor_predictor(features)
        cognitive_pred = self.cognitive_predictor(features)
        return motor_pred, cognitive_pred


class EnhancedUnifiedGIMANSystem(nn.Module):
    """Enhanced unified GIMAN system with advanced architecture and training."""

    def __init__(self, config: EnhancedConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.embed_dim

        # Enhanced unified attention module
        self.unified_attention = EnhancedUnifiedAttentionModule(
            config.embed_dim, config.num_heads, config.dropout_rate
        )

        # Enhanced ensemble predictors
        self.ensemble_predictor = EnhancedEnsemblePredictor(
            config.embed_dim, config.embed_dim, config.dropout_rate
        )

        # Enhanced interpretability modules
        self.feature_importance = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.embed_dim // 2, config.embed_dim),
            nn.Sigmoid(),
        )

        # Counterfactual generation with enhanced architecture
        self.counterfactual_generator = nn.Sequential(
            nn.Linear(config.embed_dim + 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Enhanced weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        spatial_emb: torch.Tensor,
        genomic_emb: torch.Tensor,
        temporal_emb: torch.Tensor,
        return_attention: bool = False,
    ):
        """Enhanced forward pass with optional attention weights."""
        # Unified attention with enhanced mechanisms
        unified_features, cross_modal_weights, self_attention_weights = (
            self.unified_attention(spatial_emb, genomic_emb, temporal_emb)
        )

        # Ensemble predictions
        motor_pred, cognitive_pred = self.ensemble_predictor(unified_features)

        if return_attention:
            return (
                motor_pred,
                cognitive_pred,
                unified_features,
                cross_modal_weights,
                self_attention_weights,
            )

        return motor_pred, cognitive_pred

    def compute_feature_importance(self, features: torch.Tensor) -> torch.Tensor:
        """Compute feature importance scores."""
        return self.feature_importance(features)

    def generate_counterfactuals(
        self, features: torch.Tensor, target_changes: torch.Tensor
    ) -> torch.Tensor:
        """Generate counterfactual explanations."""
        combined_input = torch.cat([features, target_changes], dim=1)
        return self.counterfactual_generator(combined_input)


class EnhancedPhase4SystemTrainer:
    """Enhanced trainer with advanced training strategies."""

    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self.model = EnhancedUnifiedGIMANSystem(config).to(self.device)

        logger.info(f"🚀 Enhanced Phase 4 System initialized on {self.device}")
        logger.info(
            f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def prepare_data(
        self,
        spatial_embeddings: np.ndarray,
        genomic_embeddings: np.ndarray,
        temporal_embeddings: np.ndarray,
        motor_scores: np.ndarray,
        cognitive_labels: np.ndarray,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Enhanced data preparation with better splits."""
        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(spatial_embeddings),
            torch.FloatTensor(genomic_embeddings),
            torch.FloatTensor(temporal_embeddings),
            torch.FloatTensor(motor_scores),
            torch.FloatTensor(cognitive_labels),
        )

        # Enhanced train/val/test split
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader

    def train_model(self, train_loader, val_loader, num_epochs=100):
        """Enhanced training with advanced strategies."""
        # Initialize optimizer with enhanced settings
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            verbose=True,
            min_lr=1e-7,
        )

        # Initialize loss functions
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        # Training history
        history = {
            "train_losses": [],
            "val_losses": [],
            "train_motor_losses": [],
            "val_motor_losses": [],
            "train_cognitive_losses": [],
            "val_cognitive_losses": [],
            "learning_rates": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        # Warmup scheduler
        def get_lr_warmup_factor(epoch):
            if epoch < self.config.warmup_epochs:
                return (epoch + 1) / self.config.warmup_epochs
            return 1.0

        logger.info(f"🚀 Starting enhanced training for {num_epochs} epochs...")
        logger.info(
            f"📋 Config: LR={self.config.learning_rate}, WD={self.config.weight_decay}, "
            f"Warmup={self.config.warmup_epochs}, Early stopping={self.config.early_stopping_patience}"
        )

        for epoch in range(num_epochs):
            # Apply warmup
            lr_factor = get_lr_warmup_factor(epoch)
            current_lr = self.config.learning_rate * lr_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(
                train_loader, optimizer, mse_loss, bce_loss
            )

            # Validation phase
            self.model.eval()
            val_metrics = self._validate_epoch(val_loader, mse_loss, bce_loss)

            # Record history
            history["train_losses"].append(train_metrics["total_loss"])
            history["val_losses"].append(val_metrics["total_loss"])
            history["train_motor_losses"].append(train_metrics["motor_loss"])
            history["val_motor_losses"].append(val_metrics["motor_loss"])
            history["train_cognitive_losses"].append(train_metrics["cognitive_loss"])
            history["val_cognitive_losses"].append(val_metrics["cognitive_loss"])
            history["learning_rates"].append(current_lr)

            # Learning rate scheduling (after warmup)
            if epoch >= self.config.warmup_epochs:
                scheduler.step(val_metrics["total_loss"])

            # Early stopping
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_enhanced_phase4_model.pth")
                logger.info(f"💾 New best model saved (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Logging
            if epoch % 5 == 0 or epoch < 10:
                logger.info(
                    f"Epoch {epoch:3d}: Train={train_metrics['total_loss']:.4f}, "
                    f"Val={val_metrics['total_loss']:.4f}, LR={current_lr:.6f}"
                )
                logger.info(
                    f"           Motor: {train_metrics['motor_loss']:.4f}→{val_metrics['motor_loss']:.4f}, "
                    f"Cognitive: {train_metrics['cognitive_loss']:.4f}→{val_metrics['cognitive_loss']:.4f}"
                )
                logger.info(
                    f"           Grad norm: {train_metrics.get('grad_norm', 0):.4f}"
                )

            # Early stopping check
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"🛑 Early stopping at epoch {epoch} (patience={patience_counter})"
                )
                break

        # Load best model
        self.model.load_state_dict(torch.load("best_enhanced_phase4_model.pth"))
        logger.info(
            f"✅ Enhanced training completed. Best validation loss: {best_val_loss:.4f}"
        )

        history["best_val_loss"] = best_val_loss
        history["total_epochs"] = epoch + 1

        return history

    def _train_epoch(self, train_loader, optimizer, mse_loss, bce_loss):
        """Training epoch with enhanced monitoring."""
        total_loss = 0.0
        motor_loss_sum = 0.0
        cognitive_loss_sum = 0.0
        grad_norm_sum = 0.0

        for batch_idx, (
            spatial,
            genomic,
            temporal,
            motor_scores,
            cognitive_labels,
        ) in enumerate(train_loader):
            spatial = spatial.to(self.device)
            genomic = genomic.to(self.device)
            temporal = temporal.to(self.device)
            motor_scores = motor_scores.to(self.device)
            cognitive_labels = cognitive_labels.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            motor_pred, cognitive_pred = self.model(spatial, genomic, temporal)

            # Calculate losses with adaptive weighting
            motor_loss = mse_loss(motor_pred.squeeze(), motor_scores)
            cognitive_loss = bce_loss(cognitive_pred.squeeze(), cognitive_labels)

            # Dynamic task weighting based on loss magnitudes
            motor_weight = 1.0 / (1.0 + motor_loss.item())
            cognitive_weight = 1.0 / (1.0 + cognitive_loss.item())

            # Normalize weights
            total_weight = motor_weight + cognitive_weight
            motor_weight /= total_weight
            cognitive_weight /= total_weight

            combined_loss = (
                motor_weight * motor_loss + cognitive_weight * cognitive_loss
            )

            # Backward pass
            combined_loss.backward()

            # Gradient clipping with monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_value
            )

            optimizer.step()

            # Accumulate metrics
            total_loss += combined_loss.item()
            motor_loss_sum += motor_loss.item()
            cognitive_loss_sum += cognitive_loss.item()
            grad_norm_sum += grad_norm.item()

        return {
            "total_loss": total_loss / len(train_loader),
            "motor_loss": motor_loss_sum / len(train_loader),
            "cognitive_loss": cognitive_loss_sum / len(train_loader),
            "grad_norm": grad_norm_sum / len(train_loader),
        }

    def _validate_epoch(self, val_loader, mse_loss, bce_loss):
        """Validation epoch."""
        total_loss = 0.0
        motor_loss_sum = 0.0
        cognitive_loss_sum = 0.0

        with torch.no_grad():
            for (
                spatial,
                genomic,
                temporal,
                motor_scores,
                cognitive_labels,
            ) in val_loader:
                spatial = spatial.to(self.device)
                genomic = genomic.to(self.device)
                temporal = temporal.to(self.device)
                motor_scores = motor_scores.to(self.device)
                cognitive_labels = cognitive_labels.to(self.device)

                # Forward pass
                motor_pred, cognitive_pred = self.model(spatial, genomic, temporal)

                # Calculate losses
                motor_loss = mse_loss(motor_pred.squeeze(), motor_scores)
                cognitive_loss = bce_loss(cognitive_pred.squeeze(), cognitive_labels)
                combined_loss = motor_loss + cognitive_loss

                # Accumulate metrics
                total_loss += combined_loss.item()
                motor_loss_sum += motor_loss.item()
                cognitive_loss_sum += cognitive_loss.item()

        return {
            "total_loss": total_loss / len(val_loader),
            "motor_loss": motor_loss_sum / len(val_loader),
            "cognitive_loss": cognitive_loss_sum / len(val_loader),
        }

    def evaluate_model(self, test_loader):
        """Enhanced model evaluation with comprehensive metrics."""
        self.model.eval()

        all_motor_preds = []
        all_motor_true = []
        all_cognitive_preds = []
        all_cognitive_true = []

        with torch.no_grad():
            for (
                spatial,
                genomic,
                temporal,
                motor_scores,
                cognitive_labels,
            ) in test_loader:
                spatial = spatial.to(self.device)
                genomic = genomic.to(self.device)
                temporal = temporal.to(self.device)

                motor_pred, cognitive_pred = self.model(spatial, genomic, temporal)

                all_motor_preds.extend(motor_pred.cpu().numpy().flatten())
                all_motor_true.extend(motor_scores.numpy().flatten())
                all_cognitive_preds.extend(cognitive_pred.cpu().numpy().flatten())
                all_cognitive_true.extend(cognitive_labels.numpy().flatten())

        # Calculate comprehensive metrics
        motor_r2 = r2_score(all_motor_true, all_motor_preds)
        motor_mse = mean_squared_error(all_motor_true, all_motor_preds)
        motor_rmse = np.sqrt(motor_mse)

        cognitive_auc = roc_auc_score(all_cognitive_true, all_cognitive_preds)

        return {
            "motor_r2": motor_r2,
            "motor_mse": motor_mse,
            "motor_rmse": motor_rmse,
            "cognitive_auc": cognitive_auc,
            "motor_predictions": all_motor_preds,
            "motor_true": all_motor_true,
            "cognitive_predictions": all_cognitive_preds,
            "cognitive_true": all_cognitive_true,
        }


def main():
    """Main function to run enhanced Phase 4 system."""
    logger.info("🌟 Enhanced Phase 4 GIMAN System")
    logger.info("=" * 50)

    # Initialize enhanced configuration
    config = EnhancedConfig()
    logger.info(f"📋 Configuration: {config}")

    # Load data (using Phase 3.1 integration)
    from phase3_1_real_data_integration import RealDataPhase3Integration

    integrator = RealDataPhase3Integration()
    integrator.load_real_ppmi_data()
    integrator.generate_spatiotemporal_embeddings()
    integrator.generate_genomic_embeddings()
    integrator.load_prognostic_targets()

    logger.info(f"📊 Loaded data for {len(integrator.patient_ids)} patients")

    # Prepare synthetic temporal embeddings (would be replaced with real temporal data)
    np.random.seed(42)
    temporal_embeddings = np.random.randn(
        len(integrator.patient_ids), config.embed_dim
    ).astype(np.float32)

    # Initialize enhanced trainer
    trainer = EnhancedPhase4SystemTrainer(config)

    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        integrator.spatiotemporal_embeddings,
        integrator.genomic_embeddings,
        temporal_embeddings,
        integrator.prognostic_targets[:, 0],  # Motor scores
        integrator.prognostic_targets[:, 1],  # Cognitive labels
    )

    logger.info(
        f"📊 Data prepared: Train={len(train_loader.dataset)}, "
        f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}"
    )

    # Train enhanced model
    logger.info("🚀 Starting enhanced training...")
    history = trainer.train_model(train_loader, val_loader, num_epochs=150)

    # Evaluate enhanced model
    logger.info("🧪 Evaluating enhanced model...")
    results = trainer.evaluate_model(test_loader)

    # Display results
    logger.info("🎯 Enhanced Phase 4 Results:")
    logger.info(f"   Motor R²: {results['motor_r2']:.4f}")
    logger.info(f"   Motor RMSE: {results['motor_rmse']:.4f}")
    logger.info(f"   Cognitive AUC: {results['cognitive_auc']:.4f}")

    # Save results
    results_summary = {
        "config": config.__dict__,
        "training_history": history,
        "evaluation_results": results,
        "model_parameters": sum(p.numel() for p in trainer.model.parameters()),
    }

    torch.save(results_summary, "enhanced_phase4_results.pth")
    logger.info("💾 Results saved to 'enhanced_phase4_results.pth'")

    logger.info("✅ Enhanced Phase 4 system completed successfully!")

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
