#!/usr/bin/env python3
"""Debug script to check model dimensions and layer setup."""

import sys

sys.path.append(
    "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/src"
)

import torch

from giman_pipeline.training.models import GIMANClassifier


def main():
    print("🔍 Debugging Model Dimensions")
    print("=" * 50)

    # Create model with same config as training
    model = GIMANClassifier(
        input_dim=7,
        hidden_dims=[64, 128, 64],
        output_dim=4,
        dropout_rate=0.3,
        pooling_method="concat",
        classification_level="node",
    )

    print(f"Model classification level: {model.classification_level}")
    print(f"Model backbone classification level: {model.backbone.classification_level}")
    print(f"Model hidden dims: {model.backbone.hidden_dims}")

    # Check classifier structure
    print("\n🧱 Classifier Structure:")
    for i, layer in enumerate(model.backbone.classifier):
        print(f"  Layer {i}: {layer}")

    # Test with dummy data
    print("\n🧪 Testing with dummy data:")

    # Create dummy graph data
    num_nodes = 10
    x = torch.randn(num_nodes, 7)  # 10 nodes, 7 features each
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.ones(edge_index.size(1))

    print(f"Input features shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")

    # Test backbone forward pass step by step
    print("\n🔍 Step-by-step forward pass:")

    h1 = model.backbone.conv1(x, edge_index)
    print(f"After conv1: {h1.shape}")

    h1 = model.backbone.bn1(h1)
    h1 = torch.nn.functional.relu(h1)
    h1 = model.backbone.dropout(h1)
    print(f"After processing conv1: {h1.shape}")

    h2 = model.backbone.conv2(h1, edge_index)
    print(f"After conv2: {h2.shape}")

    h2 = model.backbone.bn2(h2)
    h2 = torch.nn.functional.relu(h2)
    h2 = model.backbone.dropout(h2)
    print(f"After processing conv2: {h2.shape}")

    h3 = model.backbone.conv3(h2, edge_index, edge_weight)
    print(f"After conv3: {h3.shape}")

    h3 = model.backbone.bn3(h3)

    # Residual connection
    if model.backbone.use_residual:
        residual = (
            model.backbone.residual_proj(h1)
            if model.backbone.residual_proj is not None
            else h1
        )
        print(f"Residual shape: {residual.shape}")
        h3 = h3 + residual
        print(f"After residual: {h3.shape}")

    h3 = torch.nn.functional.relu(h3)
    h3 = model.backbone.dropout(h3)
    print(f"Final node embeddings: {h3.shape}")

    # Test classifier
    print("\n🎯 Testing classifier:")
    print(
        f"Classifier expects input of shape: [num_nodes, {model.backbone.hidden_dims[2]}]"
    )
    print(f"Node embeddings shape: {h3.shape}")

    try:
        logits = model.backbone.classifier(h3)
        print(f"✅ Classification successful! Logits shape: {logits.shape}")
    except Exception as e:
        print(f"❌ Classification failed: {e}")


if __name__ == "__main__":
    main()
