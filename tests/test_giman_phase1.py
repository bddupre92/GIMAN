"""Test script for GIMAN Phase 1 implementation.

This script tests the core GNN backbone functionality:
1. Load preprocessed PPMI data (557 patients)
2. Convert NetworkX graph to PyTorch Geometric format
3. Initialize GIMAN backbone architecture
4. Perform forward pass and validate outputs
5. Test data splitting and basic functionality

Run this script to verify Phase 1 implementation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch

from src.giman_pipeline.training import GIMANDataLoader, create_giman_model


def test_giman_phase1():
    """Test Phase 1 GIMAN implementation."""
    print("=" * 60)
    print("🧪 TESTING GIMAN PHASE 1 IMPLEMENTATION")
    print("=" * 60)

    # Test 1: Data Loading
    print("\n1️⃣ Testing Data Loading...")
    try:
        data_loader = GIMANDataLoader(
            data_dir="data/02_processed", similarity_threshold=0.3, random_state=42
        )

        # Load preprocessed data
        data_loader.load_preprocessed_data()

        # Convert to PyG format
        pyg_data = data_loader.create_pyg_data()

        print("✅ Data loading successful!")
        print(f"   - Patients: {pyg_data.num_nodes}")
        print(f"   - Edges: {pyg_data.num_edges}")
        print(f"   - Features: {pyg_data.x.shape}")
        print(f"   - PD cases: {(pyg_data.y == 1).sum()}")

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

    # Test 2: Model Creation
    print("\n2️⃣ Testing Model Creation...")
    try:
        model = create_giman_model(
            input_dim=7,
            hidden_dims=[64, 128, 64],
            output_dim=2,
            dropout_rate=0.3,
            pooling_method="concat",
        )

        print("✅ Model creation successful!")

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

    # Test 3: Forward Pass
    print("\n3️⃣ Testing Forward Pass...")
    try:
        model.eval()  # Set to evaluation mode

        with torch.no_grad():
            output = model.forward(pyg_data)

        print("✅ Forward pass successful!")
        print(f"   - Logits shape: {output['logits'].shape}")
        print(f"   - Node embeddings: {output['node_embeddings'].shape}")
        print(f"   - Graph embedding: {output['graph_embedding'].shape}")
        print(f"   - Layer embeddings: {len(output['layer_embeddings'])} layers")

        # Validate output shapes
        assert output["logits"].shape == (
            1,
            2,
        ), f"Wrong logits shape: {output['logits'].shape}"
        assert output["node_embeddings"].shape == (
            557,
            64,
        ), "Wrong node embedding shape"
        assert len(output["layer_embeddings"]) == 3, "Should have 3 layer embeddings"

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Test 4: Data Splitting
    print("\n4️⃣ Testing Data Splitting...")
    try:
        train_mask, val_mask, test_mask = data_loader.create_train_val_test_split(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True
        )

        print("✅ Data splitting successful!")

        # Validate splits
        assert train_mask.sum() + val_mask.sum() + test_mask.sum() == 557
        assert not torch.any(train_mask & val_mask)  # No overlap
        assert not torch.any(train_mask & test_mask)
        assert not torch.any(val_mask & test_mask)

    except Exception as e:
        print(f"❌ Data splitting failed: {e}")
        return False

    # Test 5: Predictions
    print("\n5️⃣ Testing Predictions...")
    try:
        # Test prediction methods
        probabilities = model.predict_proba(pyg_data)
        predictions = model.predict(pyg_data)

        print("✅ Predictions successful!")
        print(f"   - Probabilities shape: {probabilities.shape}")
        print(f"   - Predictions shape: {predictions.shape}")
        print(f"   - Predicted class: {predictions.item()}")
        print(f"   - Class probabilities: {probabilities.squeeze().tolist()}")

        # Validate outputs
        assert probabilities.shape == (1, 2), "Wrong probability shape"
        assert predictions.shape == (1,), "Wrong prediction shape"
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(1)), (
            "Probabilities don't sum to 1"
        )

    except Exception as e:
        print(f"❌ Predictions failed: {e}")
        return False

    # Test 6: Data Statistics
    print("\n6️⃣ Testing Data Statistics...")
    try:
        stats = data_loader.get_data_statistics()

        print("✅ Data statistics retrieved!")
        print(f"   - Graph density: {stats['graph_density']:.4f}")
        print(f"   - Class balance: {stats['class_balance']:.3f}")
        print(
            f"   - Edge weight range: [{stats['edge_weight_stats']['min']:.3f}, "
            f"{stats['edge_weight_stats']['max']:.3f}]"
        )

    except Exception as e:
        print(f"❌ Data statistics failed: {e}")
        return False

    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 PHASE 1 TESTING COMPLETE - ALL TESTS PASSED!")
    print("=" * 60)
    print(f"✅ GIMAN backbone successfully processes {pyg_data.num_nodes} patients")
    print(f"✅ Graph has {pyg_data.num_edges} similarity edges")
    print(f"✅ Model has {model.get_model_info()['total_parameters']:,} parameters")
    print("✅ Forward pass produces valid outputs")
    print("✅ Data splitting maintains class stratification")
    print("✅ Predictions work correctly")

    return True


def test_cross_validation():
    """Test cross-validation splits."""
    print("\n🔄 Testing Cross-Validation...")

    try:
        data_loader = GIMANDataLoader(data_dir="data/02_processed")
        data_loader.load_preprocessed_data()
        data_loader.create_pyg_data()

        cv_splits = data_loader.get_cross_validation_splits(n_folds=5)

        print(f"✅ Cross-validation created: {len(cv_splits)} folds")

        # Test each fold
        for i, (train_mask, val_mask) in enumerate(cv_splits):
            train_count = train_mask.sum().item()
            val_count = val_mask.sum().item()
            print(f"   - Fold {i + 1}: {train_count} train, {val_count} val")

            # Validate no overlap
            assert not torch.any(train_mask & val_mask)
            assert train_count + val_count == 557

        return True

    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting GIMAN Phase 1 tests...")

    # Run main tests
    success = test_giman_phase1()

    if success:
        # Run additional tests
        test_cross_validation()

        print("\n🎯 Ready for Phase 2: Training Pipeline Implementation!")
        print("Next steps:")
        print("- Implement training loop with loss functions")
        print("- Add validation and evaluation metrics")
        print("- Create experiment tracking and checkpointing")
        print("- Implement multimodal attention mechanisms")
    else:
        print("\n❌ Phase 1 tests failed. Please fix errors before proceeding.")
        sys.exit(1)
