"""
GIMIN graph construction subpackage.

Provides patient similarity graph construction from partially observed
multimodal clinical features. The key innovation is computing pairwise
cosine similarity using only mutually observed features, solving the
chicken-and-egg problem where features are needed to build the graph
but the graph is needed to impute features.

Modules:
    partial_similarity: Core partial-observation graph builder.
    modality_subgraphs: Per-modality subgraph construction.
    incremental: Online patient addition and graph management.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from gimin.graph.incremental import IncrementalGraphManager
from gimin.graph.modality_subgraphs import ModalitySubgraphBuilder
from gimin.graph.partial_similarity import PartialObservationGraphBuilder

__all__ = [
    "PartialObservationGraphBuilder",
    "ModalitySubgraphBuilder",
    "IncrementalGraphManager",
    "build_patient_graph",
]


def build_patient_graph(
    features_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    k_neighbors: int = 15,
    min_overlap: int = 3,
    similarity_metric: str = "cosine",
) -> dict[str, Any]:
    """Build a patient graph from DataFrames.

    Patients with fewer than ``min_overlap`` observed features are
    filtered out before computing pairwise similarity (they can never
    form edges). Edge indices in the result refer to rows in the
    *filtered* subset. The key ``eligible_indices`` maps filtered row
    positions back to the original DataFrame row positions.

    Args:
        features_df: Feature DataFrame (N, D). NaN = missing.
        mask_df: Binary observation mask DataFrame (N, D).
        k_neighbors: Number of nearest neighbors.
        min_overlap: Minimum shared observed features for an edge.
        similarity_metric: Similarity function (only 'cosine' supported).

    Returns:
        Dictionary with ``edge_index``, ``edge_weight``, ``overlap_frac``,
        ``num_nodes``, ``num_edges``, ``num_isolated``,
        ``eligible_indices`` (original row indices of eligible patients),
        and ``total_patients`` (original N).
    """
    import logging

    logger = logging.getLogger(__name__)

    mask_np = mask_df.values.astype(np.float32)
    obs_counts = mask_np.sum(axis=1)
    eligible = obs_counts >= min_overlap
    eligible_idx = np.where(eligible)[0]

    logger.info(
        "Filtering to patients with >= %d observed features: %d / %d eligible (%.1f%%)",
        min_overlap,
        len(eligible_idx),
        len(mask_df),
        100 * len(eligible_idx) / max(len(mask_df), 1),
    )

    features_np = features_df.fillna(0).values.astype(np.float32)
    feat_sub = features_np[eligible_idx]
    mask_sub = mask_np[eligible_idx]

    builder = PartialObservationGraphBuilder(
        k_neighbors=k_neighbors,
        min_overlap=min_overlap,
        similarity_metric=similarity_metric,
    )
    result = builder.build_full_graph(feat_sub, mask_sub)
    result["eligible_indices"] = eligible_idx
    result["total_patients"] = len(mask_df)
    return result
