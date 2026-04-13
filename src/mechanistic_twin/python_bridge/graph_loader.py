"""Minimal Python helper to load Paper 3 Graph-DT kNN graph from PyTorch checkpoint.

graph_loader.py — called from Julia (via PythonCall) to load the
Paper 3 Graph-DT checkpoint and return the kNN graph as plain numpy
arrays. We avoid `paper3.graph_digital_twin.load_graph_dt_checkpoint`
because:

1. It auto-selects MPS/CUDA device (deep review flag) — we want CPU.
2. It instantiates the full PyTorch model, which we don't need.

We only need the four graph artifacts saved alongside the model
weights: edge_index, edge_weight, pat_to_gidx, node_baseline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_paper3_graph(checkpoint_path: str) -> dict:
    """Load just the graph metadata from a Paper 3 Graph-DT checkpoint.

    Returns a dict with plain numpy arrays + a python dict — all of
    which PythonCall converts cleanly to Julia types.

    Keys returned:
        edge_index    : (2, n_edges) int64 ndarray
        edge_weight   : (n_edges,) float32 ndarray
        node_baseline : (n_nodes, n_baseline_features) float32 ndarray
        patnos        : (n_nodes,) int64 ndarray (PATNO at each graph index)
        k_neighbors   : int (=15 for fold0)
        n_nodes       : int
        n_edges       : int
        fold_idx      : int
    """
    cp = torch.load(
        checkpoint_path, weights_only=False, map_location=torch.device("cpu")
    )

    edge_index = cp["edge_index"].cpu().numpy().astype(np.int64)
    edge_weight = cp["edge_weight"].cpu().numpy().astype(np.float32)
    node_baseline = cp["node_baseline"].cpu().numpy().astype(np.float32)

    # pat_to_gidx is {PATNO: graph_index}; invert and sort by graph index.
    pat_to_gidx = cp["pat_to_gidx"]
    n_nodes = len(pat_to_gidx)
    patnos = np.zeros(n_nodes, dtype=np.int64)
    for patno, gidx in pat_to_gidx.items():
        patnos[gidx] = int(patno)

    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "node_baseline": node_baseline,
        "patnos": patnos,
        "k_neighbors": int(cp["k_neighbors"]),
        "n_nodes": int(n_nodes),
        "n_edges": int(edge_index.shape[1]),
        "fold_idx": int(cp["fold_idx"]),
    }


def neighbors_for_patno(graph: dict, patno: int) -> np.ndarray:
    """Return the array of PATNOs that are kNN-neighbors of `patno`.

    Given the graph dict from `load_paper3_graph` and a PATNO, return
    the array of PATNOs that are kNN-neighbors of `patno`. The Paper 3
    graph is undirected (edges duplicated both directions in edge_index).
    """
    patnos = graph["patnos"]
    edge_index = graph["edge_index"]

    # Find this patient's graph index
    matches = np.where(patnos == int(patno))[0]
    if len(matches) == 0:
        return np.array([], dtype=np.int64)
    gidx = int(matches[0])

    # Find all edges where src == gidx; collect the dst graph indices.
    src_mask = edge_index[0] == gidx
    neighbor_gidxs = edge_index[1, src_mask]
    return patnos[neighbor_gidxs]


if __name__ == "__main__":
    # Smoke test from the command line
    import json

    repo_root = Path(__file__).resolve().parents[3]
    ckpt = (
        repo_root / "outputs" / "paper3_checkpoints" / "graph_dt" / "fold0_graph_dt.pt"
    )
    g = load_paper3_graph(str(ckpt))
    print(
        json.dumps(
            {
                "n_nodes": g["n_nodes"],
                "n_edges": g["n_edges"],
                "k_neighbors": g["k_neighbors"],
                "fold_idx": g["fold_idx"],
                "patnos_first5": g["patnos"][:5].tolist(),
                "edge_index_shape": list(g["edge_index"].shape),
            },
            indent=2,
        )
    )
    # Pick a PATNO and show its neighbors
    test_patno = int(g["patnos"][0])
    nbrs = neighbors_for_patno(g, test_patno)
    print(f"\nNeighbors of PATNO {test_patno}: {nbrs.tolist()}")
