#!/usr/bin/env python3
"""GIMAN Integration Test - CNN + GRU Spatiotemporal Embeddings
==========================================================

Test integration of spatiotemporal embeddings with GIMAN pipeline.
Validates embedding loading, compatibility, and basic functionality.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

# Add GIMAN pipeline to path (adjust as needed)
# sys.path.append('/path/to/giman/src')


def test_embedding_loading():
    """Test loading spatiotemporal embeddings."""
    print("🧠 Testing Spatiotemporal Embedding Loading")
    print("-" * 50)

    try:
        from giman_pipeline.spatiotemporal_embeddings import (
            get_embedding_info,
            get_spatiotemporal_embedding,
        )

        # Get embedding info
        info = get_embedding_info()

        print("✅ Loaded embedding provider")
        print(f"📐 Embedding dimension: {info['metadata']['embedding_dim']}")
        print(f"🔢 Number of sessions: {info['num_sessions']}")
        print(f"👥 Available patients: {info['available_patients']}")

        # Test individual embedding retrieval
        patient_id = info["available_patients"][0]
        embedding = get_spatiotemporal_embedding(patient_id, "baseline")

        if embedding is not None:
            print(f"\n✅ Retrieved embedding for {patient_id}_baseline")
            print(f"   Shape: {embedding.shape}")
            print(f"   Mean: {np.mean(embedding):.6f}")
            print(f"   Std: {np.std(embedding):.6f}")
            print(f"   Norm: {np.linalg.norm(embedding):.6f}")

        return True

    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False


def test_embedding_quality():
    """Test quality and distribution of spatiotemporal embeddings."""
    print("\n📊 Testing Embedding Quality")
    print("-" * 50)

    try:
        from giman_pipeline.spatiotemporal_embeddings import (
            get_all_spatiotemporal_embeddings,
        )

        embeddings = get_all_spatiotemporal_embeddings()

        # Convert to array for analysis
        embedding_matrix = np.array(list(embeddings.values()))

        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        print("Global statistics:")
        print(f"  Mean: {np.mean(embedding_matrix):.6f}")
        print(f"  Std: {np.std(embedding_matrix):.6f}")
        print(f"  Min: {np.min(embedding_matrix):.6f}")
        print(f"  Max: {np.max(embedding_matrix):.6f}")

        # Check for reasonable distribution
        mean_abs = np.mean(np.abs(embedding_matrix))
        print(f"  Mean absolute value: {mean_abs:.6f}")

        if 0.001 < mean_abs < 0.1:
            print("✅ Embedding distribution looks reasonable")
        else:
            print("⚠️  Embedding distribution may need attention")

        # Create distribution plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(embedding_matrix.flatten(), bins=50, alpha=0.7)
        plt.title("Embedding Value Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 2)
        embedding_norms = [np.linalg.norm(emb) for emb in embedding_matrix]
        plt.hist(embedding_norms, bins=20, alpha=0.7)
        plt.title("Embedding Norm Distribution")
        plt.xlabel("L2 Norm")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 3)
        embedding_means = [np.mean(emb) for emb in embedding_matrix]
        plt.hist(embedding_means, bins=20, alpha=0.7)
        plt.title("Per-Embedding Mean Distribution")
        plt.xlabel("Mean Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(
            "spatiotemporal_embedding_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("📈 Saved embedding analysis plot: spatiotemporal_embedding_analysis.png")

        return True

    except Exception as e:
        print(f"❌ Error analyzing embeddings: {e}")
        return False


def test_patient_consistency():
    """Test consistency of embeddings within patients."""
    print("\n👥 Testing Patient Embedding Consistency")
    print("-" * 50)

    try:
        from giman_pipeline.spatiotemporal_embeddings import spatiotemporal_provider

        available_patients = spatiotemporal_provider.get_available_patients()

        for patient_id in available_patients:
            patient_embeddings = spatiotemporal_provider.get_patient_embeddings(
                patient_id
            )

            if len(patient_embeddings) > 1:
                # Calculate similarity between sessions for this patient
                sessions = list(patient_embeddings.keys())
                emb1 = patient_embeddings[sessions[0]]
                emb2 = patient_embeddings[sessions[1]]

                # Cosine similarity
                cosine_sim = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )

                # L2 distance
                l2_dist = np.linalg.norm(emb1 - emb2)

                print(f"Patient {patient_id}: {len(patient_embeddings)} sessions")
                print(f"  Cosine similarity: {cosine_sim:.4f}")
                print(f"  L2 distance: {l2_dist:.4f}")

                # Reasonable similarity expected for same patient
                if 0.7 < cosine_sim < 0.99:
                    print("  ✅ Reasonable inter-session similarity")
                else:
                    print("  ⚠️  Unusual inter-session similarity")

        return True

    except Exception as e:
        print(f"❌ Error testing patient consistency: {e}")
        return False


def main():
    """Run complete integration test."""
    print("\n" + "=" * 60)
    print("🚀 GIMAN SPATIOTEMPORAL EMBEDDING INTEGRATION TEST")
    print("=" * 60)

    tests = [
        ("Embedding Loading", test_embedding_loading),
        ("Embedding Quality", test_embedding_quality),
        ("Patient Consistency", test_patient_consistency),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))

    # Summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All tests passed! Spatiotemporal embeddings ready for GIMAN!")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Please check integration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
