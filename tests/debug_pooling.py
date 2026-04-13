#!/usr/bin/env python3
"""Test to understand the pooling dimension calculation."""


def test_pooling_calculation():
    node_embed_dim = 64
    pooling_method = "concat"

    if pooling_method == "mean" or pooling_method == "max":
        pooled_dim = node_embed_dim
    elif pooling_method == "concat":
        pooled_dim = node_embed_dim * 2  # Mean + Max concatenation
    else:
        raise ValueError(f"Unsupported pooling method: {pooling_method}")

    print(f"Node embedding dim: {node_embed_dim}")
    print(f"Pooling method: {pooling_method}")
    print(f"Pooled dim: {pooled_dim}")

    classification_level = "node"
    hidden_dims = [64, 128, 64]

    print(f"\nClassification level: {classification_level}")
    print(f"hidden_dims[2]: {hidden_dims[2]}")

    if classification_level == "node":
        input_dim = hidden_dims[2]  # Should be 64
        print(f"Node-level input dim: {input_dim}")
    else:
        input_dim = pooled_dim  # Should be 128 (but only for graph-level)
        print(f"Graph-level input dim: {input_dim}")


if __name__ == "__main__":
    test_pooling_calculation()
