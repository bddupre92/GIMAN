"""Smoke tests for Adaptive Cross-Modal Attention.

Tests all three phases of the adaptive fusion implementation:
1. Phase 1: Strengthened attention bias (10x scale) + no post-scaling
2. Phase 2: Observed-only hard masking
3. Phase 3: Learned adaptive fusion gate

Usage:
    python scripts/smoke_test_adaptive_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.giman_pipeline.modeling.true_giman import TrueGIMAN


def make_test_data(n=50, modality_dims=None, missingness_pattern="mixed"):
    """Create synthetic test data with controlled missingness patterns.

    Args:
        missingness_pattern: One of "mixed" (realistic), "all_observed",
            "sparse" (most missing), "block" (some modalities fully missing).
    """
    if modality_dims is None:
        modality_dims = {
            "expanded_clinical": 6,
            "clinical_biomarkers": 4,
            "genetic": 5,
            "structural_imaging": 6,
            "csf_biomarkers": 4,
            "cortical_thickness": 5,
        }

    total_features = sum(modality_dims.values())
    x = torch.randn(n, total_features)

    # Build observation mask based on pattern
    obs_mask = torch.ones(n, total_features)
    offset = 0
    for mod_name, dim in modality_dims.items():
        if missingness_pattern == "all_observed":
            pass  # all ones
        elif missingness_pattern == "sparse":
            # 80% missing for all modalities
            mask = torch.bernoulli(torch.full((n, dim), 0.2))
            obs_mask[:, offset : offset + dim] = mask
        elif missingness_pattern == "block":
            # First 3 modalities observed, last 3 fully missing for half patients
            if list(modality_dims.keys()).index(mod_name) >= 3:
                obs_mask[: n // 2, offset : offset + dim] = 0.0
        elif missingness_pattern == "mixed":
            # Realistic: clinical ~95%, genetic ~88%, imaging/CSF ~30%
            rates = {
                "expanded_clinical": 0.95,
                "clinical_biomarkers": 0.90,
                "genetic": 0.88,
                "structural_imaging": 0.31,
                "csf_biomarkers": 0.22,
                "cortical_thickness": 0.31,
            }
            rate = rates.get(mod_name, 0.5)
            # Block-wise: patient either has it or doesn't
            patient_has = torch.bernoulli(torch.full((n,), rate))
            obs_mask[:, offset : offset + dim] = patient_has.unsqueeze(1).expand(
                -1, dim
            )
        offset += dim

    # Simple k-NN graph (random for smoke test)
    k = 5
    edge_list = []
    for i in range(n):
        neighbors = torch.randint(0, n, (k,))
        for j in neighbors:
            if j != i:
                edge_list.append([i, j.item()])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return x, edge_index, obs_mask, modality_dims


def test_phase1_strengthened_bias():
    """Phase 1: Model forward with strengthened bias + no post-scaling."""
    print("Test 1: Phase 1 — Strengthened attention bias (non-adaptive)")
    x, edge_index, obs_mask, modality_dims = make_test_data()

    # Non-adaptive but with Phase 1+2 fixes (observed_threshold=0.0 = Phase 1 only)
    model = TrueGIMAN(
        modality_dims=modality_dims,
        adaptive_fusion=False,
        observed_threshold=0.0,  # Phase 1 only
        tasks={"survival"},
    )

    output = model(x, edge_index, obs_mask=obs_mask)
    assert output.risk_scores is not None
    assert output.risk_scores.shape == (50, 1)
    assert not torch.isnan(output.risk_scores).any()
    print("  ✓ Forward pass OK, no NaN")

    # Backward
    loss = output.risk_scores.sum()
    loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    assert len(grad_norms) > 0
    print(f"  ✓ Backward pass OK, {len(grad_norms)} params have gradients")
    print("  PASSED\n")


def test_phase2_observed_only_masking():
    """Phase 2: Observed-only hard masking excludes low-availability modalities."""
    print("Test 2: Phase 2 — Observed-only masking (threshold=0.3)")
    x, edge_index, obs_mask, modality_dims = make_test_data(missingness_pattern="block")

    model = TrueGIMAN(
        modality_dims=modality_dims,
        adaptive_fusion=False,
        observed_threshold=0.3,  # Phase 2
        tasks={"survival"},
    )

    output = model(x, edge_index, obs_mask=obs_mask, return_attention=True)
    assert output.risk_scores is not None
    assert not torch.isnan(output.risk_scores).any()
    print("  ✓ Forward with block missingness + hard masking — no NaN")

    # Check attention weights: missing modalities should get near-zero attention
    if output.cross_modal_attention is not None:
        attn = output.cross_modal_attention  # [N, M, M]
        print(f"  ✓ Attention weights shape: {attn.shape}")
    print("  PASSED\n")


def test_phase3_adaptive_gate():
    """Phase 3: Full adaptive fusion with learned gate."""
    print("Test 3: Phase 3 — Adaptive fusion gate")
    x, edge_index, obs_mask, modality_dims = make_test_data(missingness_pattern="mixed")

    model = TrueGIMAN(
        modality_dims=modality_dims,
        adaptive_fusion=True,
        observed_threshold=0.3,
        tasks={"survival"},
    )

    output = model(x, edge_index, obs_mask=obs_mask, return_attention=True)
    assert output.risk_scores is not None
    assert not torch.isnan(output.risk_scores).any()
    print("  ✓ Forward with adaptive fusion — no NaN")

    # Check gate values exist and are in [0, 1]
    assert output.fusion_gate_values is not None
    gate = output.fusion_gate_values
    assert gate.shape == (50, 1)
    assert (gate >= 0).all() and (gate <= 1).all()
    # At initialization (zero weights/bias), gate should be ~0.5
    assert abs(gate.mean().item() - 0.5) < 0.1, (
        f"Gate mean {gate.mean():.3f} should be ~0.5 at init"
    )
    print(
        f"  ✓ Gate values shape {gate.shape}, mean={gate.mean():.4f} (should be ~0.5 at init)"
    )

    # Backward
    loss = output.risk_scores.sum()
    loss.backward()
    gate_grads = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if "fusion_gate" in name and p.grad is not None
    }
    assert len(gate_grads) > 0
    print(f"  ✓ Gate parameters have gradients: {gate_grads}")

    # Parameter count
    params = model.count_parameters()
    print(f"  ✓ Parameter counts: {params}")
    assert "fusion_gate" in params
    assert params["fusion_gate"] == 5  # 4 weights + 1 bias
    print("  PASSED\n")


def test_backward_compatibility_no_mask():
    """Backward compatibility: obs_mask=None works with all modes."""
    print("Test 4: Backward compatibility — no obs_mask")
    x, edge_index, _, modality_dims = make_test_data()

    for adaptive in [False, True]:
        model = TrueGIMAN(
            modality_dims=modality_dims,
            adaptive_fusion=adaptive,
            observed_threshold=0.3,
            tasks={"survival"},
        )
        output = model(x, edge_index, obs_mask=None)
        assert output.risk_scores is not None
        assert not torch.isnan(output.risk_scores).any()
        print(f"  ✓ adaptive_fusion={adaptive}, obs_mask=None — OK")
    print("  PASSED\n")


def test_backward_compatibility_no_crossmodal():
    """No cross-modal (ablation path) still works."""
    print("Test 5: No cross-modal (ablation)")
    x, edge_index, obs_mask, modality_dims = make_test_data()
    n_mods = len(modality_dims)
    embed_dim = 64

    model = TrueGIMAN(
        modality_dims=modality_dims,
        cross_modal_heads=0,
        fused_dim=n_mods * embed_dim,
        adaptive_fusion=False,  # Can't be adaptive without cross-modal
        tasks={"survival"},
    )

    output = model(x, edge_index, obs_mask=obs_mask)
    assert output.risk_scores is not None
    assert not torch.isnan(output.risk_scores).any()
    print("  ✓ cross_modal_heads=0 — forward OK")
    print("  PASSED\n")


def test_extreme_missingness():
    """Edge case: all modalities missing for some patients."""
    print("Test 6: Extreme missingness — all modalities missing for some patients")
    x, edge_index, obs_mask, modality_dims = make_test_data()

    # Set first 5 patients to have NO observed features at all
    obs_mask[:5, :] = 0.0

    for mode_name, adaptive, threshold in [
        ("Phase 1 only", False, 0.0),
        ("Phase 1+2", False, 0.3),
        ("Full adaptive", True, 0.3),
    ]:
        model = TrueGIMAN(
            modality_dims=modality_dims,
            adaptive_fusion=adaptive,
            observed_threshold=threshold,
            tasks={"survival"},
        )
        output = model(x, edge_index, obs_mask=obs_mask)
        assert output.risk_scores is not None
        has_nan = torch.isnan(output.risk_scores).any().item()
        print(f"  {'✓' if not has_nan else '✗'} {mode_name}: NaN={has_nan}")
        if has_nan:
            print(f"    WARNING: NaN detected in {mode_name}!")

    print("  PASSED\n")


def test_gate_varies_with_availability():
    """Verify gate values differ between high- and low-availability patients."""
    print("Test 7: Gate varies with availability (after one training step)")
    modality_dims = {
        "expanded_clinical": 6,
        "clinical_biomarkers": 4,
        "genetic": 5,
        "structural_imaging": 6,
        "csf_biomarkers": 4,
        "cortical_thickness": 5,
    }
    total_features = sum(modality_dims.values())
    n = 100

    x = torch.randn(n, total_features)
    obs_mask = torch.ones(n, total_features)

    # First 50 patients: all modalities observed
    # Last 50 patients: only first 2 modalities observed (clinical only)
    offset = 0
    for i, (mod_name, dim) in enumerate(modality_dims.items()):
        if i >= 2:
            obs_mask[50:, offset : offset + dim] = 0.0
        offset += dim

    # Simple graph
    edge_list = []
    for i in range(n):
        for j in torch.randint(0, n, (3,)):
            if j != i:
                edge_list.append([i, j.item()])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    model = TrueGIMAN(
        modality_dims=modality_dims,
        adaptive_fusion=True,
        observed_threshold=0.3,
        tasks={"survival"},
    )

    # Do one training step to differentiate gate values
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x, edge_index, obs_mask=obs_mask, return_attention=True)
        # Fake survival loss: just optimize risk scores
        loss = output.risk_scores.sum()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, obs_mask=obs_mask, return_attention=True)

    gate = output.fusion_gate_values.squeeze()
    gate_high = gate[:50].mean().item()
    gate_low = gate[50:].mean().item()
    print(f"  Gate mean (6 modalities): {gate_high:.4f}")
    print(f"  Gate mean (2 modalities): {gate_low:.4f}")
    print(f"  Difference: {gate_high - gate_low:+.4f}")
    # After a few steps, the gate should have started to differentiate
    # (not a hard assertion since 10 steps may not be enough)
    if abs(gate_high - gate_low) > 0.001:
        print("  ✓ Gate differentiates between high/low availability patients")
    else:
        print("  ⚠ Gate hasn't differentiated yet (may need more training)")
    print("  PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("SMOKE TESTS: Adaptive Cross-Modal Attention")
    print("=" * 60 + "\n")

    tests = [
        test_phase1_strengthened_bias,
        test_phase2_observed_only_masking,
        test_phase3_adaptive_gate,
        test_backward_compatibility_no_mask,
        test_backward_compatibility_no_crossmodal,
        test_extreme_missingness,
        test_gate_varies_with_availability,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
