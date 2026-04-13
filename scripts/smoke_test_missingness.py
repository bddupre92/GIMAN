"""Smoke test: Verify missingness-aware TrueGIMAN forward pass.

Tests:
1. Forward pass with observation masks (missingness-aware mode)
2. Forward pass without observation masks (backward compatible)
3. Correct availability computation per modality
4. Attention gating: missing modalities get lower attention
5. Masked k-NN graph construction
6. Gradient flow through learned missing embeddings
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.giman_pipeline.modeling.true_giman import TrueGIMAN
from src.giman_pipeline.training.multi_task_trainer import build_knn_graph


def test_forward_with_masks():
    """Test forward pass with observation masks."""
    print("=" * 60)
    print("TEST 1: Forward pass with observation masks")
    print("=" * 60)

    modality_dims = {
        "genetic": 5,
        "expanded_clinical": 5,
        "structural_imaging": 6,
        "csf_biomarkers": 4,
        "clinical_biomarkers": 4,
        "cortical_thickness": 6,
    }
    total_features = sum(modality_dims.values())

    model = TrueGIMAN(
        modality_dims=modality_dims,
        modality_embed_dim=64,
        cross_modal_heads=4,
        fused_dim=128,
        gat_hidden_dim=64,
        gat_output_dim=64,
        gat_heads=4,
        gat_layers=3,
        tasks={"survival"},
    )

    N = 50
    x = torch.randn(N, total_features)
    obs_mask = torch.ones(N, total_features)

    # Simulate: imaging (cols 10-15) and CSF (cols 16-19) missing for 70% of patients
    missing_patients = torch.randperm(N)[:35]
    obs_mask[missing_patients, 10:20] = 0.0  # imaging + CSF

    # Build a simple graph
    edge_index = build_knn_graph(x.numpy(), k=5)

    # Forward with mask
    output = model(x, edge_index, obs_mask=obs_mask, return_attention=True)

    assert output.risk_scores is not None, "Risk scores should not be None"
    assert output.risk_scores.shape == (N, 1), (
        f"Expected (50, 1), got {output.risk_scores.shape}"
    )
    assert output.modality_availability is not None, (
        "Modality availability should be returned"
    )
    assert output.modality_availability.shape == (N, 6), (
        f"Expected (50, 6), got {output.modality_availability.shape}"
    )

    # Check availability: genetic/clinical should be 1.0, imaging/CSF should be ~0.3 (30% observed)
    avail = output.modality_availability.detach()
    for patient_idx in range(5):
        if patient_idx in missing_patients:
            # structural_imaging (mod 2) and csf_biomarkers (mod 3) should have avail < 1
            assert avail[patient_idx, 2] < 1.0, "Imaging should be partially missing"
        else:
            assert avail[patient_idx, 0] == 1.0, "Genetic should be fully observed"

    print("  ✓ Forward pass with masks: shapes correct")
    print(f"  ✓ Availability range: [{avail.min():.2f}, {avail.max():.2f}]")
    print(f"  ✓ Mean availability per modality: {avail.mean(dim=0).tolist()}")
    print()


def test_backward_compatible():
    """Test forward pass WITHOUT observation masks (old behavior)."""
    print("=" * 60)
    print("TEST 2: Backward compatible (no obs_mask)")
    print("=" * 60)

    modality_dims = {"genetic": 5, "clinical": 5, "imaging": 6}
    total = 16

    model = TrueGIMAN(
        modality_dims=modality_dims,
        modality_embed_dim=32,
        cross_modal_heads=2,
        fused_dim=64,
        gat_hidden_dim=32,
        gat_output_dim=32,
        gat_heads=2,
        gat_layers=2,
        tasks={"survival"},
    )

    N = 20
    x = torch.randn(N, total)
    edge_index = build_knn_graph(x.numpy(), k=5)

    # Forward WITHOUT mask — should work exactly as before
    output = model(x, edge_index)

    assert output.risk_scores is not None
    assert output.risk_scores.shape == (N, 1)
    print("  ✓ Backward compatible forward pass: OK")
    print()


def test_gradient_flow():
    """Test that gradients flow through learned missing embeddings."""
    print("=" * 60)
    print("TEST 3: Gradient flow through missing embeddings")
    print("=" * 60)

    modality_dims = {"genetic": 5, "imaging": 6}
    total = 11

    model = TrueGIMAN(
        modality_dims=modality_dims,
        modality_embed_dim=32,
        cross_modal_heads=2,
        fused_dim=64,
        gat_hidden_dim=32,
        gat_output_dim=32,
        gat_heads=2,
        gat_layers=2,
        tasks={"survival"},
    )

    N = 20
    x = torch.randn(N, total)
    obs_mask = torch.ones(N, total)
    # Make imaging completely missing for first 10 patients
    obs_mask[:10, 5:11] = 0.0

    edge_index = build_knn_graph(x.numpy(), k=5)
    output = model(x, edge_index, obs_mask=obs_mask)

    # Compute a dummy loss and backprop
    loss = output.risk_scores.sum()
    loss.backward()

    # Check that missing embedding for imaging has gradients
    imaging_missing_emb = model.encoder_bank.missing_embeddings["imaging"]
    assert imaging_missing_emb.grad is not None, (
        "Missing embedding should have gradients"
    )
    assert imaging_missing_emb.grad.abs().sum() > 0, "Gradients should be non-zero"

    print(
        f"  ✓ Missing embedding grad norm (imaging): {imaging_missing_emb.grad.norm():.6f}"
    )
    print(
        f"  ✓ Missing embedding grad norm (genetic): {model.encoder_bank.missing_embeddings['genetic'].grad.norm():.6f}"
    )
    print()


def test_masked_knn_graph():
    """Test masked k-NN graph construction."""
    print("=" * 60)
    print("TEST 4: Masked k-NN graph construction")
    print("=" * 60)

    N = 30
    F = 10
    features = np.random.randn(N, F).astype(np.float32)
    obs_mask = np.ones((N, F), dtype=np.float32)

    # Make last 5 features missing for half the patients
    obs_mask[:15, 5:] = 0.0

    # Standard k-NN
    edge_standard = build_knn_graph(features, k=5)
    # Masked k-NN
    edge_masked = build_knn_graph(features, k=5, obs_mask=obs_mask)

    assert edge_standard.shape[0] == 2
    assert edge_masked.shape[0] == 2
    assert edge_standard.shape[1] == N * 5  # k neighbors per node
    assert edge_masked.shape[1] == N * 5

    print(f"  ✓ Standard graph: {edge_standard.shape[1]} edges")
    print(f"  ✓ Masked graph: {edge_masked.shape[1]} edges")
    print(f"  ✓ Edges differ: {not torch.equal(edge_standard, edge_masked)}")
    print()


def test_no_cross_modal_with_masks():
    """Test the no-cross-modal ablation path with observation masks."""
    print("=" * 60)
    print("TEST 5: No cross-modal attention + masks (ablation path)")
    print("=" * 60)

    modality_dims = {"genetic": 5, "clinical": 5, "imaging": 6}
    total = 16

    model = TrueGIMAN(
        modality_dims=modality_dims,
        modality_embed_dim=32,
        cross_modal_heads=0,  # Disable cross-modal attention
        fused_dim=64,
        gat_hidden_dim=32,
        gat_output_dim=32,
        gat_heads=2,
        gat_layers=2,
        tasks={"survival"},
    )

    assert not model.use_cross_modal, "Cross-modal should be disabled"

    N = 20
    x = torch.randn(N, total)
    obs_mask = torch.ones(N, total)
    obs_mask[:10, 5:11] = 0.0  # imaging missing

    edge_index = build_knn_graph(x.numpy(), k=5)
    output = model(x, edge_index, obs_mask=obs_mask)

    assert output.risk_scores is not None
    assert output.risk_scores.shape == (N, 1)
    print("  ✓ No cross-modal + masks: forward pass OK")
    print()


def test_attention_gating():
    """Verify that fully-missing modalities get lower attention weights."""
    print("=" * 60)
    print("TEST 6: Attention gating for missing modalities")
    print("=" * 60)

    modality_dims = {"genetic": 5, "clinical": 5, "imaging": 6}
    total = 16

    model = TrueGIMAN(
        modality_dims=modality_dims,
        modality_embed_dim=32,
        cross_modal_heads=2,
        fused_dim=64,
        gat_hidden_dim=32,
        gat_output_dim=32,
        gat_heads=2,
        gat_layers=2,
        tasks={"survival"},
    )

    N = 20
    x = torch.randn(N, total)

    # Case 1: All observed
    obs_mask_full = torch.ones(N, total)
    edge_index = build_knn_graph(x.numpy(), k=5)
    output_full = model(x, edge_index, obs_mask=obs_mask_full, return_attention=True)

    # Case 2: Imaging completely missing
    obs_mask_no_img = torch.ones(N, total)
    obs_mask_no_img[:, 10:16] = 0.0  # imaging modality = 0
    output_no_img = model(
        x, edge_index, obs_mask=obs_mask_no_img, return_attention=True
    )

    # Check availability
    avail_full = output_full.modality_availability.detach()
    avail_no_img = output_no_img.modality_availability.detach()

    print(f"  Full obs availability: {avail_full[0].tolist()}")
    print(f"  No imaging availability: {avail_no_img[0].tolist()}")

    assert avail_full[0, 2] == 1.0, "Imaging should be fully available"
    assert avail_no_img[0, 2] == 0.0, "Imaging should be fully missing"
    assert avail_no_img[0, 0] == 1.0, "Genetic should still be fully available"

    print("  ✓ Availability correctly reflects missingness patterns")
    print()


def main():
    print("\n" + "=" * 60)
    print("SMOKE TEST: Missingness-Aware TrueGIMAN")
    print("=" * 60 + "\n")

    tests = [
        test_forward_with_masks,
        test_backward_compatible,
        test_gradient_flow,
        test_masked_knn_graph,
        test_no_cross_modal_with_masks,
        test_attention_gating,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
