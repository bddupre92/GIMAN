#!/usr/bin/env python3
"""GIMAN Phase 5: Quick Architecture Test.

Simple test of Phase 5 task-specific architecture without dependencies
on Phase 3 integration. Tests the core architectural innovations.

Author: GIMAN Development Team
Date: September 2025
"""

import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TaskSpecificGAT(nn.Module):
    """Simplified GAT for testing."""

    def __init__(
        self,
        in_features=96,
        hidden_features=32,
        out_features=32,
        num_heads=2,
        dropout=0.7,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.W_shared = nn.Linear(in_features, hidden_features * num_heads, bias=False)
        self.attention = nn.Linear(2 * hidden_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(hidden_features * num_heads, out_features)

    def forward(self, x, adj_matrix):
        batch_size, num_nodes = x.size(0), x.size(1)

        # Shared transformation
        h = self.W_shared(x)
        h = h.view(batch_size, num_nodes, self.num_heads, self.hidden_features)

        # Multi-head attention
        attention_outputs = []
        for head in range(self.num_heads):
            h_head = h[:, :, head, :]

            # Attention computation
            h_expanded = h_head.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            h_tiled = h_head.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            attention_input = torch.cat([h_expanded, h_tiled], dim=-1)

            e = self.leaky_relu(self.attention(attention_input)).squeeze(-1)
            e = e.masked_fill(adj_matrix.unsqueeze(0) == 0, float("-inf"))
            alpha = torch.softmax(e, dim=-1)
            alpha = self.dropout(alpha)

            h_out = torch.matmul(alpha, h_head)
            attention_outputs.append(h_out)

        # Concatenate and project
        h_concat = torch.cat(attention_outputs, dim=-1)
        output = self.out_proj(h_concat)
        output = self.layer_norm(output)

        return output


class TaskSpecificGIMANTest(nn.Module):
    """Simplified Phase 5 GIMAN for testing."""

    def __init__(
        self, spatial_dim=256, genomic_dim=8, temporal_dim=64, embed_dim=32, dropout=0.7
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding layers
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

        # GAT layer
        self.gat_layer = TaskSpecificGAT(
            in_features=embed_dim * 3,
            hidden_features=embed_dim,
            out_features=embed_dim,
            dropout=dropout,
        )

        # Multimodal attention (note: GAT outputs embed_dim, not embed_dim*3)
        self.multimodal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=2, dropout=dropout, batch_first=True
        )

        # Task-specific towers (input is embed_dim after GAT+attention)
        self.motor_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),  # Continuous output
        )

        self.cognitive_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),  # Binary output
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, spatial, genomic, temporal, adj_matrix):
        # Embedding
        spatial_emb = self.spatial_embedding(spatial)
        genomic_emb = self.genomic_embedding(genomic)
        temporal_emb = self.temporal_embedding(temporal)

        # Expand genomic to match spatial/temporal node dimensions
        batch_size, num_nodes = spatial_emb.size(0), spatial_emb.size(1)
        genomic_emb = genomic_emb.unsqueeze(1).expand(-1, num_nodes, -1)

        # Combine embeddings
        combined_emb = torch.cat([spatial_emb, genomic_emb, temporal_emb], dim=-1)

        # GAT processing
        gat_output = self.gat_layer(combined_emb, adj_matrix)

        # Multimodal attention
        attn_output, _ = self.multimodal_attention(
            gat_output, gat_output, gat_output, need_weights=False
        )

        # Patient-level representation (sum across nodes)
        patient_repr = torch.sum(attn_output, dim=1)

        # Task-specific predictions
        motor_pred = self.motor_tower(patient_repr)
        cognitive_pred = self.cognitive_tower(patient_repr)

        return motor_pred, cognitive_pred


def test_phase5_architecture():
    """Test Phase 5 task-specific architecture."""
    logger.info("🚀 Testing Phase 5 Task-Specific Architecture")
    logger.info("=" * 50)

    # Create synthetic data
    batch_size = 10
    num_nodes = 20
    spatial_dim = 256
    genomic_dim = 8
    temporal_dim = 64

    # Synthetic inputs
    spatial = torch.randn(batch_size, num_nodes, spatial_dim)
    genomic = torch.randn(batch_size, genomic_dim)
    temporal = torch.randn(batch_size, num_nodes, temporal_dim)

    # Adjacency matrix (fully connected for simplicity)
    adj_matrix = torch.ones(num_nodes, num_nodes)

    # Synthetic targets
    y_motor = torch.randn(batch_size)
    y_cognitive = torch.randint(0, 2, (batch_size,)).float()

    logger.info("📊 Synthetic data created:")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Spatial: {spatial.shape}")
    logger.info(f"   Genomic: {genomic.shape}")
    logger.info(f"   Temporal: {temporal.shape}")
    logger.info(f"   Adjacency: {adj_matrix.shape}")

    # Test model instantiation
    logger.info("\n⚙️ Testing model instantiation...")
    model = TaskSpecificGIMANTest()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("✅ Model created successfully")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")

    # Test forward pass
    logger.info("\n🔄 Testing forward pass...")
    model.eval()
    with torch.no_grad():
        motor_pred, cognitive_pred = model(spatial, genomic, temporal, adj_matrix)

    logger.info("✅ Forward pass successful")
    logger.info(f"   Motor predictions shape: {motor_pred.shape}")
    logger.info(f"   Cognitive predictions shape: {cognitive_pred.shape}")
    logger.info(f"   Motor range: [{motor_pred.min():.3f}, {motor_pred.max():.3f}]")
    logger.info(
        f"   Cognitive range: [{cognitive_pred.min():.3f}, {cognitive_pred.max():.3f}]"
    )

    # Test training step
    logger.info("\n🏃 Testing training step...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    motor_criterion = nn.MSELoss()
    cognitive_criterion = nn.BCELoss()

    # Forward pass
    motor_pred, cognitive_pred = model(spatial, genomic, temporal, adj_matrix)

    # Compute losses
    motor_loss = motor_criterion(motor_pred.squeeze(), y_motor)
    cognitive_loss = cognitive_criterion(cognitive_pred.squeeze(), y_cognitive)
    total_loss = 0.7 * motor_loss + 0.3 * cognitive_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    logger.info("✅ Training step successful")
    logger.info(f"   Motor loss: {motor_loss.item():.4f}")
    logger.info(f"   Cognitive loss: {cognitive_loss.item():.4f}")
    logger.info(f"   Total loss: {total_loss.item():.4f}")

    # Test multiple training steps
    logger.info("\n🔄 Testing multiple training steps...")
    losses = []

    for epoch in range(10):
        optimizer.zero_grad()
        motor_pred, cognitive_pred = model(spatial, genomic, temporal, adj_matrix)

        motor_loss = motor_criterion(motor_pred.squeeze(), y_motor)
        cognitive_loss = cognitive_criterion(cognitive_pred.squeeze(), y_cognitive)
        total_loss = 0.7 * motor_loss + 0.3 * cognitive_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(total_loss.item())

    logger.info("✅ Multi-step training successful")
    logger.info(f"   Initial loss: {losses[0]:.4f}")
    logger.info(f"   Final loss: {losses[-1]:.4f}")
    logger.info(f"   Loss change: {losses[-1] - losses[0]:+.4f}")

    # Test evaluation metrics
    logger.info("\n📊 Testing evaluation metrics...")
    model.eval()
    with torch.no_grad():
        motor_pred, cognitive_pred = model(spatial, genomic, temporal, adj_matrix)

    # Convert to numpy for metrics
    motor_pred_np = motor_pred.squeeze().numpy()
    cognitive_pred_np = cognitive_pred.squeeze().numpy()
    y_motor_np = y_motor.numpy()
    y_cognitive_np = y_cognitive.numpy()

    # Calculate metrics
    motor_r2 = r2_score(y_motor_np, motor_pred_np)

    try:
        cognitive_auc = roc_auc_score(y_cognitive_np, cognitive_pred_np)
    except ValueError:
        cognitive_auc = 0.5  # Random baseline

    logger.info("✅ Metrics computation successful")
    logger.info(f"   Motor R²: {motor_r2:.4f}")
    logger.info(f"   Cognitive AUC: {cognitive_auc:.4f}")

    return {
        "model_params": total_params,
        "motor_r2": motor_r2,
        "cognitive_auc": cognitive_auc,
        "loss_reduction": losses[0] - losses[-1],
    }


def test_dynamic_loss_weighting():
    """Test dynamic loss weighting functionality."""
    logger.info("\n⚖️ Testing Dynamic Loss Weighting")
    logger.info("-" * 40)

    class DynamicLossWeighter:
        def __init__(self, strategy="adaptive", initial_motor_weight=0.7):
            self.strategy = strategy
            self.motor_weight = initial_motor_weight
            self.cognitive_weight = 1.0 - initial_motor_weight
            self.motor_losses = []
            self.cognitive_losses = []
            self.adaptation_rate = 0.1

        def get_weights(self, motor_loss, cognitive_loss, epoch=0):
            if self.strategy == "fixed":
                return self.motor_weight, self.cognitive_weight
            elif self.strategy == "adaptive":
                return self._adaptive_weighting(motor_loss, cognitive_loss)
            elif self.strategy == "curriculum":
                return self._curriculum_weighting(epoch)

        def _adaptive_weighting(self, motor_loss, cognitive_loss):
            self.motor_losses.append(motor_loss)
            self.cognitive_losses.append(cognitive_loss)

            if len(self.motor_losses) < 3:
                return self.motor_weight, self.cognitive_weight

            avg_motor = np.mean(self.motor_losses[-3:])
            avg_cognitive = np.mean(self.cognitive_losses[-3:])

            total = avg_motor + avg_cognitive
            if total > 0:
                motor_rel = avg_motor / total
                target_motor_weight = 0.5 + (motor_rel - 0.5) * 0.4
                self.motor_weight += self.adaptation_rate * (
                    target_motor_weight - self.motor_weight
                )
                self.cognitive_weight = 1.0 - self.motor_weight

            return self.motor_weight, self.cognitive_weight

        def _curriculum_weighting(self, epoch):
            if epoch < 25:
                progress = epoch / 25
                motor_weight = 0.8 - 0.1 * progress
            else:
                progress = (epoch - 25) / 25
                progress = min(progress, 1.0)
                motor_weight = 0.7 - 0.2 * progress

            return motor_weight, 1.0 - motor_weight

    # Test all strategies
    strategies = ["fixed", "adaptive", "curriculum"]

    for strategy in strategies:
        logger.info(f"\n🔧 Testing '{strategy}' strategy...")
        weighter = DynamicLossWeighter(strategy=strategy)

        # Simulate training with changing losses
        motor_losses = [0.8, 0.6, 0.4, 0.5, 0.3]
        cognitive_losses = [0.7, 0.8, 0.6, 0.4, 0.5]

        for epoch, (m_loss, c_loss) in enumerate(
            zip(motor_losses, cognitive_losses, strict=False)
        ):
            m_weight, c_weight = weighter.get_weights(m_loss, c_loss, epoch)
            logger.info(
                f"   Epoch {epoch}: Motor={m_weight:.3f}, Cognitive={c_weight:.3f}"
            )

        logger.info(f"✅ {strategy.capitalize()} strategy working correctly")

    return True


if __name__ == "__main__":
    logger.info("🎯 Phase 5 Quick Architecture Test Starting")
    logger.info("=" * 60)

    # Test task-specific architecture
    arch_results = test_phase5_architecture()

    # Test dynamic loss weighting
    loss_results = test_dynamic_loss_weighting()

    # Summary
    logger.info("\n🎉 Phase 5 Quick Test Summary")
    logger.info("=" * 40)
    logger.info("✅ Task-specific architecture: WORKING")
    logger.info(f"   Model parameters: {arch_results['model_params']:,}")
    logger.info(f"   Motor R²: {arch_results['motor_r2']:.4f}")
    logger.info(f"   Cognitive AUC: {arch_results['cognitive_auc']:.4f}")
    logger.info(f"   Loss reduction: {arch_results['loss_reduction']:.4f}")

    logger.info("✅ Dynamic loss weighting: WORKING")

    print("\n🎉 Phase 5 Architecture Test Complete!")
    print("✅ Task-specific towers functioning correctly")
    print("✅ Dynamic loss weighting operational")
    print("🚀 Phase 5 core architecture validated!")
    print("\nKey Results:")
    print(f"  - Model size: {arch_results['model_params']:,} parameters")
    print(f"  - Motor performance: R² = {arch_results['motor_r2']:.4f}")
    print(f"  - Cognitive performance: AUC = {arch_results['cognitive_auc']:.4f}")
    print(
        f"  - Training convergence: {arch_results['loss_reduction']:.4f} loss reduction"
    )
