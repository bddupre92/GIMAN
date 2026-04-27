#!/usr/bin/env python3
"""
Research Plan Phase 2, Task 2.1: GIMANPrognostic Model with Dual Prediction Heads

This implements the prognostic GIMAN architecture specified in the research plan:
- Dual-task learning: Motor progression (regression) + Cognitive decline (classification)
- Graph Attention Network (GAT) backbone
- Cross-modal attention fusion (prepared for multimodal encoders)
- Separate prediction heads for each task

Author: GIMAN Development Team
Date: October 2, 2025
Research Plan Phase: 2 (Prognostic Model Architecture)
Task: 2.1 - Create GIMANPrognostic class with dual prediction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GIMANPrognostic(nn.Module):
    """
    Graph-Informed Multimodal Attention Network for Parkinson's Disease Progression Prediction.

    This model implements dual-task prognostic prediction:
    1. Motor Progression: Regression of UPDRS-III slope (continuous)
    2. Cognitive Decline: Classification of MCI conversion (binary)

    Architecture:
        Input Features → GAT Layers → Cross-Modal Attention → Dual Prediction Heads

    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        num_gat_layers: Number of Graph Attention layers
        num_attention_heads: Number of attention heads in GAT
        dropout: Dropout probability
        use_cross_modal_attention: Whether to use cross-modal attention (for future multimodal integration)

    Research Plan Alignment:
        - Stage III architecture (Graph-Attention Fusion + Dual Heads)
        - Designed to integrate with Stage II encoders (spatiotemporal, genomic, clinical)
        - Currently uses baseline features; ready for multimodal encoder integration
    """

    def __init__(
        self,
        input_dim: int = 7,  # Baseline: 7 biomarker features from Phase 1
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.3,
        use_cross_modal_attention: bool = False
    ):
        super(GIMANPrognostic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.use_cross_modal_attention = use_cross_modal_attention

        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Graph Attention Network layers
        self.gat_layers = nn.ModuleList()

        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_attention_heads,
                heads=num_attention_heads,
                dropout=dropout,
                concat=True
            )
        )

        # Intermediate GAT layers
        for _ in range(num_gat_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_attention_heads,
                    heads=num_attention_heads,
                    dropout=dropout,
                    concat=True
                )
            )

        # Final GAT layer (average heads instead of concat)
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_attention_heads,
                dropout=dropout,
                concat=False  # Average instead of concatenate
            )
        )

        # Layer normalization after each GAT layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
        ])

        # Cross-modal attention (for future multimodal integration)
        if use_cross_modal_attention:
            self.cross_modal_attention = CrossModalAttention(
                feature_dim=hidden_dim,
                num_modalities=3,  # Spatiotemporal, Genomic, Clinical
                num_heads=num_attention_heads
            )

        # Shared feature extraction
        self.shared_features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task-specific prediction heads

        # Motor Progression Head (Regression)
        self.motor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  # Single continuous output (UPDRS-III slope)
        )

        # Cognitive Decline Head (Binary Classification)
        self.cognitive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)  # Binary classification (decline vs stable)
        )

        logger.info(f"GIMANPrognostic initialized:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  GAT layers: {num_gat_layers}")
        logger.info(f"  Attention heads: {num_attention_heads}")
        logger.info(f"  Cross-modal attention: {use_cross_modal_attention}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GIMANPrognostic.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
            return_embeddings: If True, return intermediate embeddings for interpretability

        Returns:
            motor_pred: Motor progression predictions [num_nodes, 1] (continuous)
            cognitive_pred: Cognitive decline predictions [num_nodes, 2] (logits)

        Research Plan Note:
            - Motor prediction: UPDRS-III slope (points/year) - will use MSE loss
            - Cognitive prediction: Binary MCI conversion - will use Focal Loss
        """
        # Input projection
        h = self.input_projection(x)

        # Graph Attention Network layers
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat_layer(h, edge_index)
            h_new = layer_norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connection (skip connection)
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new

        # Cross-modal attention (if enabled - for future multimodal integration)
        if self.use_cross_modal_attention:
            # Currently using single modality; will integrate spatiotemporal/genomic/clinical later
            h = self.cross_modal_attention(h)

        # Shared feature extraction
        shared_features = self.shared_features(h)

        # Task-specific prediction heads
        motor_pred = self.motor_head(shared_features)  # [num_nodes, 1]
        cognitive_pred = self.cognitive_head(shared_features)  # [num_nodes, 2]

        if return_embeddings:
            return motor_pred, cognitive_pred, shared_features

        return motor_pred, cognitive_pred

    def get_attention_weights(self, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention weights from specified GAT layer for interpretability.

        Args:
            layer_idx: Index of GAT layer (default: -1 for last layer)

        Returns:
            Attention weights from the specified layer
        """
        # This will be populated during forward pass
        # Requires modifying GATConv to return attention weights
        # Implementation for Task 6.1 (Interpretability)
        raise NotImplementedError("Attention weight extraction will be implemented in Phase 6")


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing multiple modalities.

    This module is prepared for Research Plan Stage II encoder integration:
    - Spatiotemporal imaging features (3D CNN-GRU)
    - Genomic features (Transformer)
    - Clinical trajectory features (GRU)

    Currently not used in baseline model but ready for Phase 3-4 integration.
    """

    def __init__(
        self,
        feature_dim: int,
        num_modalities: int = 3,
        num_heads: int = 4
    ):
        super(CrossModalAttention, self).__init__()

        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Modality-specific projections (for future use)
        self.modality_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(num_modalities)
        ])

        logger.info(f"CrossModalAttention initialized for {num_modalities} modalities")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modal attention.

        Args:
            features: Input features [num_nodes, feature_dim]

        Returns:
            Fused features [num_nodes, feature_dim]

        Note:
            Currently single-modality passthrough.
            Will be expanded in Research Plan Phase 3 (Multimodal Feature Integration)
        """
        # Placeholder for single modality
        # In Phase 3, this will fuse spatiotemporal, genomic, and clinical features
        return features


class ModelSummary:
    """Utility class for printing model architecture summary."""

    @staticmethod
    def print_summary(model: GIMANPrognostic, input_dim: int = 7):
        """
        Print detailed model summary.

        Args:
            model: GIMANPrognostic model instance
            input_dim: Input feature dimension
        """
        print("="*80)
        print("GIMAN PROGNOSTIC MODEL ARCHITECTURE SUMMARY")
        print("="*80)
        print(f"\nResearch Plan Phase 2, Task 2.1")
        print(f"Dual-Task Prognostic Prediction Model\n")

        print("INPUT")
        print(f"  Node features: [num_nodes, {input_dim}]")
        print(f"  Edge index: [2, num_edges]")
        print(f"  Edge weights: [num_edges] (optional)\n")

        print("ARCHITECTURE")
        print(f"  1. Input Projection: {input_dim} -> {model.hidden_dim}")
        print(f"  2. GAT Layers: {model.num_gat_layers} layers")
        print(f"     - Attention heads: {model.num_attention_heads}")
        print(f"     - Hidden dim: {model.hidden_dim}")
        print(f"     - Dropout: {model.dropout}")
        print(f"  3. Cross-Modal Attention: {model.use_cross_modal_attention}")
        print(f"  4. Shared Features: {model.hidden_dim} -> {model.hidden_dim}")
        print(f"  5. Task-Specific Heads:")
        print(f"     - Motor Head: {model.hidden_dim} -> {model.hidden_dim//2} -> {model.hidden_dim//4} -> 1")
        print(f"     - Cognitive Head: {model.hidden_dim} -> {model.hidden_dim//2} -> {model.hidden_dim//4} -> 2\n")

        print("OUTPUT")
        print(f"  Motor progression: [num_nodes, 1] (continuous - UPDRS-III slope)")
        print(f"  Cognitive decline: [num_nodes, 2] (logits - binary classification)\n")

        print("PARAMETERS")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (float32)\n")

        print("RESEARCH PLAN ALIGNMENT")
        print("  [OK] Dual prediction heads (motor + cognitive)")
        print("  [OK] Graph Attention Network backbone")
        print("  [OK] Prepared for cross-modal attention")
        print("  [OK] Ready for Phase 1 data integration (2,046 patients)")
        print("  [NEXT] Future: Integrate spatiotemporal/genomic/clinical encoders (Phase 3-4)")
        print("="*80)


def create_giman_prognostic(
    input_dim: int = 7,
    hidden_dim: int = 128,
    num_gat_layers: int = 3,
    num_attention_heads: int = 4,
    dropout: float = 0.3,
    use_cross_modal_attention: bool = False
) -> GIMANPrognostic:
    """
    Factory function to create GIMANPrognostic model.

    Args:
        input_dim: Input feature dimension (default: 7 for baseline features)
        hidden_dim: Hidden layer dimension
        num_gat_layers: Number of GAT layers
        num_attention_heads: Number of attention heads
        dropout: Dropout probability
        use_cross_modal_attention: Enable cross-modal attention

    Returns:
        GIMANPrognostic model instance
    """
    model = GIMANPrognostic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_gat_layers=num_gat_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        use_cross_modal_attention=use_cross_modal_attention
    )

    return model


if __name__ == '__main__':
    """
    Test script for GIMANPrognostic model.

    This demonstrates:
    1. Model creation
    2. Forward pass with dummy data
    3. Output shapes for dual-task predictions
    """

    print("\n" + "="*80)
    print("RESEARCH PLAN PHASE 2, TASK 2.1: GIMAN PROGNOSTIC MODEL TEST")
    print("="*80 + "\n")

    # Create model
    model = create_giman_prognostic(
        input_dim=7,  # Phase 1 baseline features
        hidden_dim=128,
        num_gat_layers=3,
        num_attention_heads=4,
        dropout=0.3,
        use_cross_modal_attention=False
    )

    # Print model summary
    ModelSummary.print_summary(model, input_dim=7)

    # Test forward pass with dummy data
    print("\nTEST FORWARD PASS")
    print("-"*80)

    # Create dummy data
    num_nodes = 100  # 100 patients
    num_edges = 600  # k=6 nearest neighbors
    input_dim = 7

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  edge_index: {edge_index.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        motor_pred, cognitive_pred = model(x, edge_index)

    print(f"\nOutput shapes:")
    print(f"  Motor progression: {motor_pred.shape} (continuous)")
    print(f"  Cognitive decline: {cognitive_pred.shape} (logits)")

    print(f"\nSample predictions (first 5 patients):")
    print(f"  Motor slopes (UPDRS-III pts/year):")
    for i in range(5):
        print(f"    Patient {i}: {motor_pred[i, 0].item():.3f}")

    print(f"\n  Cognitive decline probabilities:")
    cognitive_probs = torch.softmax(cognitive_pred, dim=1)
    for i in range(5):
        print(f"    Patient {i}: Stable={cognitive_probs[i, 0].item():.3f}, Decline={cognitive_probs[i, 1].item():.3f}")

    print("\n" + "="*80)
    print("[OK] TASK 2.1 COMPLETE: GIMANPrognostic model successfully created")
    print("="*80)
    print("\nNext steps:")
    print("  - Task 2.2: Implement multi-task loss function (MSE + Focal Loss)")
    print("  - Task 2.3: Create training pipeline for dual-task learning")
    print("  - Task 2.4: Implement evaluation metrics (MAE/R2, AUC/F1)")
    print("  - Task 2.5: Hyperparameter tuning framework")
    print("="*80 + "\n")
