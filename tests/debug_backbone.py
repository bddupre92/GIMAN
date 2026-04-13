#!/usr/bin/env python3
"""Simple test to create fresh model and check dimensions."""

import sys

sys.path.append(
    "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/src"
)

import torch

from giman_pipeline.training.models import GIMANBackbone


def main():
    print("🔍 Testing GIMANBackbone directly")
    print("=" * 40)

    # Create backbone directly
    backbone = GIMANBackbone(
        input_dim=7,
        hidden_dims=[64, 128, 64],
        output_dim=4,
        dropout_rate=0.3,
        pooling_method="concat",
        classification_level="node",
    )

    print(f"Backbone classification level: {backbone.classification_level}")
    print(f"Backbone hidden dims: {backbone.hidden_dims}")

    # Check classifier structure
    print("\n🧱 Backbone Classifier Structure:")
    for i, layer in enumerate(backbone.classifier):
        print(f"  Layer {i}: {layer}")

    # Test forward pass
    print("\n🧪 Testing forward pass:")

    num_nodes = 10
    x = torch.randn(num_nodes, 7)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.ones(edge_index.size(1))

    try:
        output = backbone.forward(x, edge_index, edge_weight)
        logits = output["logits"]
        print("✅ Forward pass successful!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Node embeddings shape: {output['node_embeddings'].shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")


if __name__ == "__main__":
    main()
