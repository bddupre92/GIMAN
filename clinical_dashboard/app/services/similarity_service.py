"""Patient similarity service: k-NN lookup from Graph-DT graph."""

from __future__ import annotations

import logging

import torch

from app.data.schema import PatientNeighbor

logger = logging.getLogger(__name__)


def find_similar_patients(
    patno: int,
    model_registry,
    patient_store,
    k: int = 10,
) -> list[PatientNeighbor]:
    """Find k most similar patients using the Graph-DT graph structure."""

    if model_registry.graphdt_model is None or model_registry.graphdt_ckpt is None:
        logger.warning("Graph-DT not loaded — cannot find similar patients")
        return []

    pat_to_gidx = model_registry.graphdt_ckpt.get("pat_to_gidx", {})
    edge_index = model_registry.graphdt_ckpt.get("edge_index")

    if edge_index is None:
        return []

    # Convert string keys to int if needed
    if pat_to_gidx and isinstance(next(iter(pat_to_gidx.keys())), str):
        pat_to_gidx = {int(k_): v for k_, v in pat_to_gidx.items()}

    if patno not in pat_to_gidx:
        logger.info(f"Patient {patno} not in Graph-DT graph (fold 0)")
        return []

    target_gidx = pat_to_gidx[patno]

    # Build reverse mapping (graph idx → patno)
    gidx_to_pat = {v: k_ for k_, v in pat_to_gidx.items()}

    # Find neighbors via edge_index
    edge_np = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index

    # Edges where target is source
    mask = edge_np[0] == target_gidx
    neighbor_gidxs = edge_np[1, mask]

    # Use node embeddings for similarity scoring
    neighbors = []
    node_enc = model_registry.graphdt_node_enc
    if node_enc is not None:
        target_emb = node_enc[target_gidx]
        for gidx in neighbor_gidxs:
            neighbor_emb = node_enc[gidx]
            similarity = float(torch.nn.functional.cosine_similarity(
                target_emb.unsqueeze(0), neighbor_emb.unsqueeze(0)
            ))
            neighbor_patno = gidx_to_pat.get(int(gidx))
            if neighbor_patno and neighbor_patno != patno:
                meta = patient_store.patient_index.get(neighbor_patno, {})
                neighbors.append(
                    PatientNeighbor(
                        patno=neighbor_patno,
                        similarity=round(similarity, 4),
                        current_stage=meta.get("current_stage", "?"),
                        n_visits=meta.get("n_visits", 0),
                        stage_trajectory=patient_store.get_patient_detail(neighbor_patno).get(
                            "stage_trajectory", []
                        ) if meta else [],
                    )
                )
    else:
        # Fallback: equal weight for all neighbors
        for gidx in neighbor_gidxs:
            neighbor_patno = gidx_to_pat.get(int(gidx))
            if neighbor_patno and neighbor_patno != patno:
                meta = patient_store.patient_index.get(neighbor_patno, {})
                neighbors.append(
                    PatientNeighbor(
                        patno=neighbor_patno,
                        similarity=1.0,
                        current_stage=meta.get("current_stage", "?"),
                        n_visits=meta.get("n_visits", 0),
                        stage_trajectory=[],
                    )
                )

    # Sort by similarity, take top k
    neighbors.sort(key=lambda n: n.similarity, reverse=True)
    return neighbors[:k]
