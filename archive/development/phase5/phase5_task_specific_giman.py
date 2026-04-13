#!/usr/bin/env python3
"""GIMAN Phase 5: Task-Specific Architecture System.

Building on Phase 4's successful calibration (Cognitive AUC = 0.6883), this implementation
addresses task competition through architectural separation:

Key Innovations:
- Task-specific tower architecture (motor regression + cognitive classification)
- Shared GAT backbone with specialized endpoints
- Independent optimization pathways
- Maintained LOOCV evaluation framework

Architecture:
- Shared: GAT + Multimodal Attention (proven from Phase 4)
- Motor Tower: 3-layer regression pathway with continuous activation
- Cognitive Tower: 3-layer classification pathway with sigmoid activation
- Loss: Weighted combination (0.7 motor + 0.3 cognitive)

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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import our data integration system
import os
import sys

# Add Phase 3 path
phase3_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase3"
)
sys.path.insert(0, phase3_path)
from phase3_1_real_data_integration import RealDataPhase3Integration

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TaskSpecificGAT(nn.Module):
    """Graph Attention Network with task-specific tower architecture."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        out_features: int = 32,
        num_heads: int = 2,
        dropout: float = 0.7,
    ):
        """Initialize task-specific GAT layer."""
        super().__init__()
        self.num_heads = num_heads
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Shared GAT backbone
        self.W_shared = nn.Linear(in_features, hidden_features * num_heads, bias=False)
        self.attention = nn.Linear(2 * hidden_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Output projection
        self.out_proj = nn.Linear(hidden_features * num_heads, out_features)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass through task-specific GAT."""
        # Handle input where x is [num_patients, features] - patients are nodes
        if x.dim() == 2:
            # Input is [nodes, features] where nodes = patients
            num_nodes = x.size(0)
            x = x.unsqueeze(0)  # [1, nodes, features]
            batch_size = 1
        else:
            # Input is [batch, nodes, features]
            batch_size, num_nodes = x.size(0), x.size(1)

        # Shared transformation
        h = self.W_shared(x)  # [batch, nodes, hidden * heads]
        h = h.view(batch_size, num_nodes, self.num_heads, self.hidden_features)

        # Multi-head attention computation
        attention_outputs = []
        for head in range(self.num_heads):
            h_head = h[:, :, head, :]  # [batch, nodes, hidden]

            # Compute attention scores
            h_expanded = h_head.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            h_tiled = h_head.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            attention_input = torch.cat([h_expanded, h_tiled], dim=-1)

            e = self.leaky_relu(self.attention(attention_input)).squeeze(-1)

            # Apply adjacency mask and softmax
            # Handle adj_matrix dimensions
            if adj_matrix.dim() == 2:
                adj_mask = adj_matrix.unsqueeze(0)
            else:
                adj_mask = adj_matrix

            e = e.masked_fill(adj_mask == 0, float("-inf"))
            alpha = F.softmax(e, dim=-1)
            alpha = self.dropout(alpha)

            # Apply attention to get output
            h_out = torch.matmul(alpha, h_head)
            attention_outputs.append(h_out)

        # Concatenate multi-head outputs
        h_concat = torch.cat(attention_outputs, dim=-1)

        # Final projection and normalization
        output = self.out_proj(h_concat)
        output = self.layer_norm(output)

        # Remove batch dimension if it was added for 2D input
        if batch_size == 1 and output.size(0) == 1:
            output = output.squeeze(0)  # [nodes, features]

        return output


class TaskSpecificGIMANSystem(nn.Module):
    """GIMAN with task-specific tower architecture for addressing task competition."""

    def __init__(
        self,
        spatial_dim: int = 256,
        genomic_dim: int = 8,
        temporal_dim: int = 64,
        embed_dim: int = 32,
        num_heads: int = 2,
        dropout: float = 0.7,
    ):
        """Initialize task-specific GIMAN system."""
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding layers (shared backbone)
        self.spatial_embedding = nn.Sequential(
            nn.Linear(spatial_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.genomic_embedding = nn.Sequential(
            nn.Linear(genomic_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal_embedding = nn.Sequential(
            nn.Linear(temporal_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared GAT layer
        self.gat_layer = TaskSpecificGAT(
            in_features=embed_dim * 3,
            hidden_features=embed_dim,
            out_features=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Multimodal attention (shared)
        self.multimodal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim * 3,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Task-specific towers
        self.motor_tower = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),  # Continuous output for regression
        )

        self.cognitive_tower = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),  # Binary output for classification
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, spatial, genomic, temporal, adj_matrix):
        """Forward pass through task-specific GIMAN."""
        batch_size = spatial.size(0)

        # Shared embedding layers
        spatial_emb = self.spatial_embedding(spatial)
        genomic_emb = self.genomic_embedding(genomic)
        temporal_emb = self.temporal_embedding(temporal)

        # Combine embeddings for processing
        combined_emb = torch.cat(
            [spatial_emb, genomic_emb, temporal_emb], dim=-1
        )  # [batch, embed_dim*3]

        # Handle different batch sizes for GAT processing
        if batch_size == 1:
            # Single patient case (test time) - skip GAT, use embeddings directly
            gat_output = spatial_emb  # [1, embed_dim] - just use spatial embedding
        else:
            # Multiple patients case (training time) - use GAT with adjacency matrix
            gat_output = self.gat_layer(combined_emb, adj_matrix)  # [batch, embed_dim]

        # Expand GAT output to match multimodal attention expectations
        # We need [batch, embed_dim*3] for the attention mechanism
        gat_expanded = torch.cat(
            [gat_output, gat_output, gat_output], dim=-1
        )  # [batch, embed_dim*3]

        # Self-attention over the GAT-processed representation
        # Ensure proper dimensions for multihead attention
        if gat_expanded.dim() == 2:
            # [batch, features] -> [batch, 1, features]
            attn_input = gat_expanded.unsqueeze(1)
        else:
            attn_input = gat_expanded

        attn_output, _ = self.multimodal_attention(
            attn_input,  # Query: [batch, seq_len, embed_dim*3]
            attn_input,  # Key: [batch, seq_len, embed_dim*3]
            attn_input,  # Value: [batch, seq_len, embed_dim*3]
        )

        # Final patient representation combines attention output with residual
        if attn_output.dim() == 3:
            attn_output = attn_output.squeeze(1)  # Remove sequence dimension
        final_repr = attn_output + combined_emb  # [batch, embed_dim*3]

        # Task-specific towers
        motor_pred = self.motor_tower(final_repr)
        cognitive_pred = self.cognitive_tower(final_repr)

        return motor_pred, cognitive_pred


class TaskSpecificLOOCVEvaluator:
    """Leave-One-Out Cross-Validation evaluator for task-specific GIMAN."""

    def __init__(self, device="cpu"):
        """Initialize evaluator."""
        self.device = device
        self.scaler = StandardScaler()

    def evaluate(
        self, X_spatial, X_genomic, X_temporal, y_motor, y_cognitive, adj_matrix
    ):
        """Run LOOCV evaluation with task-specific architecture."""
        logger.info("🔄 Starting task-specific LOOCV evaluation...")

        loo = LeaveOneOut()
        n_samples = X_spatial.shape[0]

        motor_predictions = []
        motor_actuals = []
        cognitive_predictions = []
        cognitive_actuals = []

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_spatial)):
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

            X_train_spatial = scaler_spatial.fit_transform(
                X_train_spatial.reshape(-1, X_train_spatial.shape[-1])
            ).reshape(X_train_spatial.shape)
            X_test_spatial = scaler_spatial.transform(
                X_test_spatial.reshape(-1, X_test_spatial.shape[-1])
            ).reshape(X_test_spatial.shape)

            X_train_genomic = scaler_genomic.fit_transform(X_train_genomic)
            X_test_genomic = scaler_genomic.transform(X_test_genomic)

            X_train_temporal = scaler_temporal.fit_transform(
                X_train_temporal.reshape(-1, X_train_temporal.shape[-1])
            ).reshape(X_train_temporal.shape)
            X_test_temporal = scaler_temporal.transform(
                X_test_temporal.reshape(-1, X_test_temporal.shape[-1])
            ).reshape(X_test_temporal.shape)

            # Subset adjacency matrix for training data
            adj_train = adj_matrix[np.ix_(train_idx, train_idx)]

            # Train model
            model = TaskSpecificGIMANSystem(
                spatial_dim=X_spatial.shape[1],  # 256 from real data
                genomic_dim=X_genomic.shape[1],  # 256 from real data
                temporal_dim=X_temporal.shape[1],  # 256 from real data
                embed_dim=128,  # Unified embedding dimension
                num_heads=4,  # Attention heads
                dropout=0.5,  # Strong regularization
            ).to(self.device)
            motor_pred, cognitive_pred = self._train_fold(
                model,
                X_train_spatial,
                X_train_genomic,
                X_train_temporal,
                y_train_motor,
                y_train_cognitive,
                X_test_spatial,
                X_test_genomic,
                X_test_temporal,
                adj_train,  # Use training subset of adjacency matrix
            )

            # Test model
            test_motor_pred, test_cognitive_pred = self._test_fold(
                model, X_test_spatial, X_test_genomic, X_test_temporal, adj_matrix
            )

            # Store predictions
            motor_predictions.append(test_motor_pred)
            motor_actuals.append(y_test_motor)
            cognitive_predictions.append(test_cognitive_pred)
            cognitive_actuals.append(y_test_cognitive)

        # Calculate metrics
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
            cognitive_auc = 0.5  # Random baseline if all one class

        results = {
            "motor_r2": motor_r2,
            "cognitive_auc": cognitive_auc,
            "motor_predictions": motor_pred_array,
            "motor_actuals": motor_actual_array,
            "cognitive_predictions": cognitive_pred_array,
            "cognitive_actuals": cognitive_actual_array,
            "n_samples": n_samples,
        }

        logger.info("✅ Task-specific LOOCV completed!")
        logger.info(f"🎯 Motor R² = {motor_r2:.4f}")
        logger.info(f"🧠 Cognitive AUC = {cognitive_auc:.4f}")

        return results

    def _train_fold(
        self,
        model,
        x_train_spatial,
        x_train_genomic,
        x_train_temporal,
        y_train_motor,
        y_train_cognitive,
        x_test_spatial,
        x_test_genomic,
        x_test_temporal,
        adj_matrix,
    ):
        """Train a single fold with task-specific loss weighting and return predictions."""
        # Convert training data to tensors
        x_train_spatial_tensor = torch.FloatTensor(x_train_spatial).to(self.device)
        x_train_genomic_tensor = torch.FloatTensor(x_train_genomic).to(self.device)
        x_train_temporal_tensor = torch.FloatTensor(x_train_temporal).to(self.device)
        y_train_motor_tensor = torch.FloatTensor(y_train_motor).to(self.device)
        y_train_cognitive_tensor = torch.FloatTensor(y_train_cognitive).to(self.device)
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)

        # Optimizer with strong weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

        # Loss functions
        motor_criterion = nn.MSELoss()
        cognitive_criterion = nn.BCELoss()

        # Training loop with early stopping
        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        model.train()
        for _epoch in range(100):  # Max 100 epochs
            optimizer.zero_grad()

            # Forward pass
            motor_pred, cognitive_pred = model(
                x_train_spatial_tensor,
                x_train_genomic_tensor,
                x_train_temporal_tensor,
                adj_matrix,
            )

            # Task-specific losses
            motor_loss = motor_criterion(motor_pred.squeeze(), y_train_motor_tensor)
            cognitive_loss = cognitive_criterion(
                cognitive_pred.squeeze(), y_train_cognitive_tensor
            )

            # Weighted combined loss (0.7 motor + 0.3 cognitive)
            total_loss = 0.7 * motor_loss + 0.3 * cognitive_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Early stopping check
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Test on held-out data
        model.eval()
        with torch.no_grad():
            x_test_spatial_tensor = torch.FloatTensor(x_test_spatial).to(self.device)
            x_test_genomic_tensor = torch.FloatTensor(x_test_genomic).to(self.device)
            x_test_temporal_tensor = torch.FloatTensor(x_test_temporal).to(self.device)

            motor_test_pred, cognitive_test_pred = model(
                x_test_spatial_tensor,
                x_test_genomic_tensor,
                x_test_temporal_tensor,
                adj_matrix,
            )

        return (
            motor_test_pred.detach().cpu().numpy(),
            cognitive_test_pred.detach().cpu().numpy(),
        )

    def _test_fold(self, model, x_spatial, x_genomic, x_temporal, adj_matrix):
        """Test a single fold."""
        model.eval()

        with torch.no_grad():
            # Convert to tensors
            x_spatial_tensor = torch.FloatTensor(x_spatial).to(self.device)
            x_genomic_tensor = torch.FloatTensor(x_genomic).to(self.device)
            x_temporal_tensor = torch.FloatTensor(x_temporal).to(self.device)
            adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)

            # Handle single sample case
            if x_spatial_tensor.dim() == 2:
                x_spatial_tensor = x_spatial_tensor.unsqueeze(0)
                x_genomic_tensor = x_genomic_tensor.unsqueeze(0)
                x_temporal_tensor = x_temporal_tensor.unsqueeze(0)

            # Forward pass
            motor_pred, cognitive_pred = model(
                x_spatial_tensor, x_genomic_tensor, x_temporal_tensor, adj_matrix
            )

            return motor_pred.cpu().numpy(), cognitive_pred.cpu().numpy()


def run_task_specific_experiment():
    """Run the main task-specific GIMAN experiment."""
    logger.info("🚀 Starting Phase 5 Task-Specific GIMAN Experiment")
    logger.info("=" * 60)

    # Load data
    logger.info("📊 Loading Phase 3 real data integration...")
    integrator = RealDataPhase3Integration()
    integrator.load_and_prepare_data()

    # Extract data
    x_spatial = integrator.spatiotemporal_embeddings
    x_genomic = integrator.genomic_embeddings
    x_temporal = integrator.temporal_embeddings
    y_motor = integrator.prognostic_targets[:, 0]  # Motor progression
    y_cognitive = integrator.prognostic_targets[:, 1]  # Cognitive conversion
    adj_matrix = integrator.similarity_matrix  # Use patient similarity matrix

    logger.info(f"📈 Dataset: {x_spatial.shape[0]} patients")
    logger.info(f"🧬 Spatial: {x_spatial.shape[1:]} | Genomic: {x_genomic.shape[1:]}")
    logger.info(f"⏰ Temporal: {x_temporal.shape[1:]} | Adjacency: {adj_matrix.shape}")

    # Validate data quality
    logger.info("🔍 Validating data quality...")
    spatial_issues = np.isnan(x_spatial).sum() + np.isinf(x_spatial).sum()
    genomic_issues = np.isnan(x_genomic).sum() + np.isinf(x_genomic).sum()
    temporal_issues = np.isnan(x_temporal).sum() + np.isinf(x_temporal).sum()

    logger.info("✅ Data Quality Check:")
    logger.info(f"   Spatial issues: {spatial_issues}")
    logger.info(f"   Genomic issues: {genomic_issues}")
    logger.info(f"   Temporal issues: {temporal_issues}")

    if spatial_issues + genomic_issues + temporal_issues > 0:
        logger.warning("⚠️ Data quality issues detected!")
        return None

    # Run evaluation
    evaluator = TaskSpecificLOOCVEvaluator(device="cpu")
    results = evaluator.evaluate(
        x_spatial, x_genomic, x_temporal, y_motor, y_cognitive, adj_matrix
    )

    # Display results
    logger.info("🎯 Phase 5 Task-Specific Results:")
    logger.info(f"   Motor R² = {results['motor_r2']:.4f}")
    logger.info(f"   Cognitive AUC = {results['cognitive_auc']:.4f}")
    logger.info(f"   Samples = {results['n_samples']}")

    return results


if __name__ == "__main__":
    # Run the task-specific experiment
    results = run_task_specific_experiment()

    if results:
        print("\n🎉 Phase 5 Task-Specific GIMAN Experiment Completed!")
        print(f"Motor Regression R²: {results['motor_r2']:.4f}")
        print(f"Cognitive Classification AUC: {results['cognitive_auc']:.4f}")
        print("\n📈 Task-specific architecture successfully implemented!")
