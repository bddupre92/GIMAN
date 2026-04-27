"""Unit tests for ``src/giman_pipeline/paper4/faithfulness.py`` (WS-P3-15).

Tests the edge-masking primitives + per-patient deletion-shift logic
on synthetic graphs without requiring the real Graph-DT checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_find_incoming_neighbors_simple():
    """Edge index with 4 edges → 2 with target_gidx=2, 2 with target_gidx=3."""
    from giman_pipeline.paper4.faithfulness import find_incoming_neighbors

    # Edges: 0->2, 1->2, 0->3, 1->3
    edge_index = torch.tensor([[0, 1, 0, 1], [2, 2, 3, 3]], dtype=torch.long)

    e_idx, src = find_incoming_neighbors(edge_index, target_gidx=2)
    np.testing.assert_array_equal(e_idx, [0, 1])
    np.testing.assert_array_equal(src, [0, 1])

    e_idx, src = find_incoming_neighbors(edge_index, target_gidx=3)
    np.testing.assert_array_equal(e_idx, [2, 3])
    np.testing.assert_array_equal(src, [0, 1])


def test_find_incoming_neighbors_no_match():
    """Target node with no incoming edges → empty arrays."""
    from giman_pipeline.paper4.faithfulness import find_incoming_neighbors

    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    e_idx, src = find_incoming_neighbors(edge_index, target_gidx=99)
    assert len(e_idx) == 0
    assert len(src) == 0


def test_mask_edges_drops_correct_columns():
    """Removing specified edge indices yields the right edge_index/edge_weight."""
    from giman_pipeline.paper4.faithfulness import mask_edges

    edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.long)
    edge_weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    ei2, ew2 = mask_edges(edge_index, edge_weight, drop_edge_idx=[1, 3])

    # Should have edges 0, 2, 4 remaining
    np.testing.assert_array_equal(ei2[0].numpy(), [0, 2, 4])
    np.testing.assert_array_equal(ei2[1].numpy(), [5, 7, 9])
    np.testing.assert_allclose(ew2.numpy(), [0.1, 0.3, 0.5])


def test_mask_edges_empty_drop_returns_originals():
    """Empty drop list returns the original tensors unchanged."""
    from giman_pipeline.paper4.faithfulness import mask_edges

    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.7])

    ei2, ew2 = mask_edges(edge_index, edge_weight, drop_edge_idx=[])
    assert ei2 is edge_index
    assert ew2 is edge_weight


def test_compute_faithfulness_with_dummy_model():
    """Synthetic 2-class survival model + tiny graph → faithfulness record
    has the expected shape (3 k_mask values, Spearman in [-1, 1])."""
    from giman_pipeline.paper4.faithfulness import (
        PatientFaithfulnessRecord,
        compute_faithfulness_for_patient,
    )

    # Tiny model that returns a deterministic CIF-like tensor based on graph features
    n_nodes, hidden_dim = 20, 8
    n_causes, n_time_bins = 2, 3

    class DummyGraphDT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, hidden_dim)
            self.head = torch.nn.Linear(hidden_dim, n_causes * n_time_bins + 1)
            self.n_causes = n_causes
            self.n_time_bins = n_time_bins

        def compute_graph_features(self, node_baseline, edge_index, edge_weight=None):
            # Aggregate neighbor features by mean (so masking edges shifts output)
            x = self.linear(node_baseline)  # (N, hidden_dim)
            agg = torch.zeros_like(x)
            counts = torch.zeros(x.shape[0])
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                agg[d] = agg[d] + x[s]
                counts[d] += 1
            counts = counts.clamp(min=1).unsqueeze(-1)
            return agg / counts

        def predict_cif(self, sequences, seq_lens, stage_idxs, graph_idxs, graph_node_features):
            gf = graph_node_features[graph_idxs]  # (batch, hidden_dim)
            pmf_logits = self.head(gf)  # (batch, n_causes*n_time_bins + 1)
            event_pmf = torch.softmax(pmf_logits, dim=-1)[:, :-1].view(
                -1, n_causes, n_time_bins
            )
            return torch.cumsum(event_pmf, dim=-1)

    model = DummyGraphDT()
    model.eval()

    rng = np.random.RandomState(0)
    node_baseline = torch.from_numpy(rng.randn(n_nodes, 2).astype(np.float32))
    # Build edges: node 5 has 6 incoming edges from nodes [0..5]
    src_list = [0, 1, 2, 3, 4, 6]
    dst_list = [5] * 6
    # Plus some unrelated edges so the graph is non-trivial
    for s in range(7, 15):
        for d in range(15, 19):
            src_list.append(s)
            dst_list.append(d)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(rng.uniform(0.5, 1.0, size=len(src_list)).astype(np.float32))

    sequences = torch.zeros(1, 1, 2)  # dummy
    seq_lens = torch.tensor([1])
    stage_idxs = torch.tensor([0])
    graph_idxs = torch.tensor([5])

    rec = compute_faithfulness_for_patient(
        model=model,
        patno=42,
        gidx=5,
        sequences=sequences,
        seq_lens=seq_lens,
        stage_idxs=stage_idxs,
        graph_idxs=graph_idxs,
        node_baseline=node_baseline,
        edge_index=edge_index,
        edge_weight=edge_weight,
        k_mask_values=(1, 3, 5),
        n_random_seeds=2,
        rng_seed=0,
        device=torch.device("cpu"),
    )
    assert isinstance(rec, PatientFaithfulnessRecord)
    assert rec.patno == 42
    assert rec.gidx == 5
    assert rec.n_neighbors == 6  # 6 incoming edges to node 5
    assert set(rec.top_k_shift_l1.keys()) == {1, 3, 5}
    for k in (1, 3, 5):
        assert rec.top_k_shift_l1[k] >= 0  # shifts are L1 distances
        assert rec.random_k_shift_l1_mean[k] >= 0
    assert -1.0 <= rec.spearman_attn_vs_delete <= 1.0
