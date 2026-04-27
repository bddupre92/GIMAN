"""Faithfulness metrics for the Graph-DT patient-similarity attention pathway.

WS-P3-15 (reviewer3.com): does the model's reliance on its k-NN
patient-similarity graph reflect a faithful attribution mechanism, or
are the graph edges a non-explanatory regularizer?

We test this with the standard graph-explanation deletion experiment
(Yuan et al. 2022 \"Explainability in Graph Neural Networks: A
Taxonomic Survey\"; following the comprehensiveness/sufficiency rubric
of DeYoung et al. 2020 ERASER):

For each test patient p with k-NN neighbors sorted by edge weight
(cosine similarity prior), we delete the top-k_mask edges incident to
p, recompute the GAT-enriched node features, re-predict the CIF, and
measure the L1 shift against the unmasked baseline. We then repeat the
deletion with k_mask randomly-selected neighbors and report the
``faithfulness gap`` = mean(top-k shift) − mean(random-k shift). A
positive gap demonstrates that high-edge-weight neighbors disproportionately
drive predictions; a near-zero gap suggests the kNN edges contribute
diffuse, non-prioritized regularization.

This is the appropriate Mac-doable substitute for full Koh-Liang
influence functions on a 1{,}900-patient × 78-output deep survival
model: it directly probes the explanation mechanism the Graph-DT
exposes (per-patient \"patients-like-you\" attention) without requiring
intractable Hessian-vector products on the full training trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def find_incoming_neighbors(
    edge_index: torch.Tensor,
    target_gidx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices in ``edge_index`` of edges incident to a target node,
    plus the corresponding source-node graph indices.

    Args:
        edge_index: ``(2, E)`` long tensor with row 0 = source, row 1 = dst.
        target_gidx: graph index of the patient whose incoming neighbors we want.

    Returns:
        ``(edge_idx_array, src_gidx_array)`` — both length ``k`` (15 for the
        canonical Graph-DT kNN graph). ``edge_idx_array[i]`` is the row in
        ``edge_index`` where the i-th incoming edge lives;
        ``src_gidx_array[i]`` is the source patient's graph index.
    """
    ei = edge_index.detach().cpu().numpy()
    mask = ei[1] == target_gidx
    edge_idx = np.where(mask)[0]
    src_gidx = ei[0, mask]
    return edge_idx, src_gidx


def mask_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    drop_edge_idx: np.ndarray | list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(edge_index, edge_weight)`` with the specified edges removed.

    Args:
        edge_index: ``(2, E)``.
        edge_weight: ``(E,)``.
        drop_edge_idx: indices into the ``E`` axis of ``edge_index`` to drop.

    Returns:
        New edge_index/edge_weight with those columns/entries removed.
    """
    if len(drop_edge_idx) == 0:
        return edge_index, edge_weight

    keep_mask = np.ones(edge_index.shape[1], dtype=bool)
    keep_mask[np.asarray(drop_edge_idx, dtype=int)] = False
    keep_idx = torch.from_numpy(np.where(keep_mask)[0]).long()
    return edge_index[:, keep_idx], edge_weight[keep_idx]


@dataclass
class PatientFaithfulnessRecord:
    patno: int
    gidx: int
    n_neighbors: int  # k (typically 15)
    baseline_l1: float  # ||cif||_1, normalisation reference
    # Per k_mask: [k_mask=1, 3, 5] L1 shift of the patient's CIF
    top_k_shift_l1: dict[int, float]
    random_k_shift_l1_mean: dict[int, float]
    random_k_shift_l1_std: dict[int, float]
    # Faithfulness gap = top_k - random_k_mean per k_mask
    gap_l1: dict[int, float]
    # Spearman correlation between per-neighbor edge_weight and per-neighbor delete shift
    spearman_attn_vs_delete: float


def compute_faithfulness_for_patient(
    model,
    patno: int,
    gidx: int,
    sequences: torch.Tensor,
    seq_lens: torch.Tensor,
    stage_idxs: torch.Tensor,
    graph_idxs: torch.Tensor,
    node_baseline: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    k_mask_values: tuple[int, ...] = (1, 3, 5),
    n_random_seeds: int = 3,
    rng_seed: int = 42,
    device: torch.device | None = None,
) -> PatientFaithfulnessRecord:
    """Compute the per-patient deletion-shift faithfulness record.

    The patient's CIF is predicted from a single-row batch
    (``sequences/seq_lens/stage_idxs/graph_idxs`` all length 1) under
    six edge-deletion conditions plus the baseline.

    Returns a record with ``top_k_shift_l1``, ``random_k_shift_l1_mean``,
    ``gap_l1``, and ``spearman_attn_vs_delete`` for the patient.
    """
    from scipy.stats import spearmanr

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    nb = node_baseline.to(device)
    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    seqs = sequences.to(device)
    slens = seq_lens
    sidxs = stage_idxs.to(device)
    gidxs = graph_idxs.to(device)

    # Baseline CIF (full graph)
    with torch.no_grad():
        gf_full = model.compute_graph_features(nb, ei, ew)
        cif_baseline = model.predict_cif(seqs, slens, sidxs, gidxs, gf_full)
    cif_baseline_np = cif_baseline.detach().cpu().numpy()
    baseline_l1 = float(np.abs(cif_baseline_np).sum())

    # Find incoming edges for this patient
    edge_idx_arr, _ = find_incoming_neighbors(edge_index, gidx)
    k = len(edge_idx_arr)

    # Per-edge weight (cosine similarity)
    ew_cpu = edge_weight.detach().cpu().numpy()
    edge_weights_for_neighbors = ew_cpu[edge_idx_arr]

    # Per-edge sort: descending by edge_weight (top = most-similar neighbor)
    sorted_order = np.argsort(-edge_weights_for_neighbors)
    sorted_edge_idx = edge_idx_arr[sorted_order]

    rng = np.random.RandomState(rng_seed + int(patno) % 1000)

    # Per-neighbor delete shift (for Spearman vs edge_weight) — single deletions
    per_neighbor_shift = np.zeros(k)
    for i, e_idx in enumerate(edge_idx_arr):
        ei_m, ew_m = mask_edges(edge_index, edge_weight, [int(e_idx)])
        with torch.no_grad():
            gf_m = model.compute_graph_features(nb, ei_m.to(device), ew_m.to(device))
            cif_m = model.predict_cif(seqs, slens, sidxs, gidxs, gf_m)
        per_neighbor_shift[i] = float(
            np.abs(cif_m.detach().cpu().numpy() - cif_baseline_np).sum()
        )

    # Spearman ρ between per-neighbor edge_weight and per-neighbor delete shift
    if k >= 2 and per_neighbor_shift.std() > 1e-12 and edge_weights_for_neighbors.std() > 1e-12:
        rho, _ = spearmanr(edge_weights_for_neighbors, per_neighbor_shift)
        spearman = float(rho) if not np.isnan(rho) else 0.0
    else:
        spearman = 0.0

    # Top-k vs random-k batch shifts
    top_k_shift_l1: dict[int, float] = {}
    random_k_shift_l1_mean: dict[int, float] = {}
    random_k_shift_l1_std: dict[int, float] = {}
    gap_l1: dict[int, float] = {}

    for k_mask in k_mask_values:
        if k_mask > k:
            top_k_shift_l1[k_mask] = float("nan")
            random_k_shift_l1_mean[k_mask] = float("nan")
            random_k_shift_l1_std[k_mask] = float("nan")
            gap_l1[k_mask] = float("nan")
            continue

        # Top-k: drop the k_mask highest-edge-weight neighbors
        drop_top = sorted_edge_idx[:k_mask].tolist()
        ei_t, ew_t = mask_edges(edge_index, edge_weight, drop_top)
        with torch.no_grad():
            gf_t = model.compute_graph_features(nb, ei_t.to(device), ew_t.to(device))
            cif_t = model.predict_cif(seqs, slens, sidxs, gidxs, gf_t)
        shift_top = float(
            np.abs(cif_t.detach().cpu().numpy() - cif_baseline_np).sum()
        )
        top_k_shift_l1[k_mask] = shift_top

        # Random-k: average over n_random_seeds independent random k-subsets
        rand_shifts = []
        for _ in range(n_random_seeds):
            drop_rand = rng.choice(edge_idx_arr, size=k_mask, replace=False).tolist()
            ei_r, ew_r = mask_edges(edge_index, edge_weight, drop_rand)
            with torch.no_grad():
                gf_r = model.compute_graph_features(
                    nb, ei_r.to(device), ew_r.to(device)
                )
                cif_r = model.predict_cif(seqs, slens, sidxs, gidxs, gf_r)
            rand_shifts.append(
                float(
                    np.abs(cif_r.detach().cpu().numpy() - cif_baseline_np).sum()
                )
            )
        rand_shifts_arr = np.asarray(rand_shifts)
        random_k_shift_l1_mean[k_mask] = float(rand_shifts_arr.mean())
        random_k_shift_l1_std[k_mask] = float(rand_shifts_arr.std())
        gap_l1[k_mask] = shift_top - float(rand_shifts_arr.mean())

    return PatientFaithfulnessRecord(
        patno=int(patno),
        gidx=int(gidx),
        n_neighbors=int(k),
        baseline_l1=baseline_l1,
        top_k_shift_l1=top_k_shift_l1,
        random_k_shift_l1_mean=random_k_shift_l1_mean,
        random_k_shift_l1_std=random_k_shift_l1_std,
        gap_l1=gap_l1,
        spearman_attn_vs_delete=spearman,
    )
