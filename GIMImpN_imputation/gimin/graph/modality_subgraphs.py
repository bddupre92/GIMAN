"""Per-modality subgraph construction for modality-specific message passing.

In GIMIN, each of the 8 clinical modalities (genetic, motor_clinical,
structural_imaging, spect_sbr, csf_biomarkers, clinical_biomarkers,
cortical_thickness, demographics) has its own similarity subgraph.
Patients are connected in a modality subgraph only if *both* have
observations in that modality, enabling modality-specific message
passing that respects the observation structure.

This is complementary to the unified partial-observation graph: the
unified graph captures cross-modality similarity, while per-modality
subgraphs enable focused information exchange within each data type.

Classes:
    ModalitySubgraphBuilder: Build separate kNN graphs for each modality.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from gimin.graph.partial_similarity import PartialObservationGraphBuilder

logger = logging.getLogger(__name__)


class ModalitySubgraphBuilder:
    """Build separate kNN graphs for each of the 8 modalities.

    Patients are connected in a modality subgraph only if both have
    observations in that modality. This enables modality-specific
    message passing, where patients share information relevant to a
    particular data type (e.g., only sharing imaging features with
    patients who also have imaging data).

    The subgraph for modality m is built by:
        1. Selecting patients who have at least one observed feature in
           modality m.
        2. Extracting those patients' features for modality m.
        3. Computing pairwise cosine similarity on those features.
        4. Building a kNN graph among those patients.
        5. Mapping local indices back to global patient indices.

    Args:
        modality_dims: List of integers specifying the number of features
            per modality. The order must match the column layout in the
            full feature matrix. For the default 8 GIMIN modalities:
            [5, 6, 6, 6, 4, 4, 6, 2] = 39 total features.
        k_per_modality: Number of nearest neighbors per modality
            subgraph. Can be a single int (same for all modalities) or
            a list of ints (one per modality). Default: 10.
        min_features_observed: Minimum number of features a patient must
            have observed within a modality to be included in that
            modality's subgraph. Default: 1.
        modality_names: Optional list of modality names for logging and
            output dictionary keys. If None, modalities are indexed by
            integer.

    Example::

        builder = ModalitySubgraphBuilder(
            modality_dims=[5, 6, 6, 6, 4, 4, 6, 2],
            k_per_modality=10,
            modality_names=[
                'genetic', 'motor_clinical', 'structural_imaging',
                'spect_sbr', 'csf_biomarkers', 'clinical_biomarkers',
                'cortical_thickness', 'demographics',
            ],
        )
        subgraphs = builder.build_all_subgraphs(features, mask)
        for name, sg in subgraphs.items():
            print(f"{name}: {sg['edge_index'].shape[1]} edges, "
                  f"{sg['num_active_patients']} active patients")
    """

    def __init__(
        self,
        modality_dims: list[int],
        k_per_modality: int | list[int] = 10,
        min_features_observed: int = 1,
        modality_names: list[str] | None = None,
    ) -> None:
        if not modality_dims:
            raise ValueError("modality_dims must be a non-empty list.")
        if any(d < 1 for d in modality_dims):
            raise ValueError("All modality dimensions must be >= 1.")

        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)

        # Handle k_per_modality as scalar or list
        if isinstance(k_per_modality, int):
            self.k_per_modality = [k_per_modality] * self.num_modalities
        else:
            if len(k_per_modality) != self.num_modalities:
                raise ValueError(
                    f"k_per_modality list length ({len(k_per_modality)}) "
                    f"!= num_modalities ({self.num_modalities})"
                )
            self.k_per_modality = list(k_per_modality)

        if min_features_observed < 1:
            raise ValueError(
                f"min_features_observed must be >= 1, got {min_features_observed}"
            )
        self.min_features_observed = min_features_observed

        if modality_names is not None:
            if len(modality_names) != self.num_modalities:
                raise ValueError(
                    f"modality_names length ({len(modality_names)}) "
                    f"!= num_modalities ({self.num_modalities})"
                )
            self.modality_names = list(modality_names)
        else:
            self.modality_names = [str(i) for i in range(self.num_modalities)]

        # Precompute column slice boundaries
        self._slice_starts: list[int] = []
        self._slice_ends: list[int] = []
        start = 0
        for dim in modality_dims:
            self._slice_starts.append(start)
            self._slice_ends.append(start + dim)
            start += dim
        self._total_features = start

    def _get_modality_slice(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        modality_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature and mask columns for a single modality.

        Args:
            features: Full feature matrix, shape (N, D_total).
            mask: Full observation mask, shape (N, D_total).
            modality_idx: Index of the modality to extract.

        Returns:
            Tuple of (mod_features, mod_mask), each of shape (N, D_m)
            where D_m is the dimension of modality ``modality_idx``.
        """
        s = self._slice_starts[modality_idx]
        e = self._slice_ends[modality_idx]
        return features[:, s:e], mask[:, s:e]

    def _get_active_patients(
        self,
        mod_mask: np.ndarray,
    ) -> np.ndarray:
        """Find patients with sufficient observations in a modality.

        Args:
            mod_mask: Observation mask for one modality, shape (N, D_m).

        Returns:
            1D integer array of global patient indices that have at least
            ``min_features_observed`` observed features in this modality.
        """
        num_observed = mod_mask.sum(axis=1)  # (N,)
        active = np.where(num_observed >= self.min_features_observed)[0]
        return active

    def build_subgraph(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        modality_idx: int,
    ) -> dict[str, Any]:
        """Build a kNN subgraph for a single modality.

        Only patients with at least ``min_features_observed`` observed
        features in the specified modality are included. The returned
        edge_index uses global patient indices (not local indices within
        the active subset).

        Args:
            features: Full feature matrix, shape (N, D_total).
            mask: Full observation mask, shape (N, D_total).
            modality_idx: Index of the modality to build a subgraph for.

        Returns:
            Dictionary with keys:
                - ``'edge_index'``: (2, E) long tensor with global patient
                  indices.
                - ``'edge_weight'``: (E,) float tensor.
                - ``'active_patients'``: 1D numpy array of global indices
                  for patients included in this subgraph.
                - ``'num_active_patients'``: int.
                - ``'modality_name'``: str.
                - ``'modality_dim'``: int.
                - ``'similarity_matrix'``: (N_active, N_active) numpy array
                  of pairwise similarities among active patients.
        """
        mod_name = self.modality_names[modality_idx]
        mod_dim = self.modality_dims[modality_idx]
        k = self.k_per_modality[modality_idx]

        logger.debug(
            "Building subgraph for modality '%s' (dim=%d, k=%d).", mod_name, mod_dim, k
        )

        # Extract modality-specific features and mask
        mod_features, mod_mask = self._get_modality_slice(features, mask, modality_idx)

        # Find active patients
        active = self._get_active_patients(mod_mask)

        if len(active) < 2:
            logger.warning(
                "Modality '%s': only %d active patient(s), cannot build "
                "subgraph (need >= 2). Returning empty graph.",
                mod_name,
                len(active),
            )
            return {
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_weight": torch.zeros(0, dtype=torch.float32),
                "active_patients": active,
                "num_active_patients": len(active),
                "modality_name": mod_name,
                "modality_dim": mod_dim,
                "similarity_matrix": np.zeros(
                    (len(active), len(active)), dtype=np.float32
                ),
            }

        # Extract active patients' features
        active_features = mod_features[active]  # (N_active, D_m)
        active_mask = mod_mask[active]  # (N_active, D_m)

        # Within a single modality, min_overlap can be 1 (all features
        # are of the same type), and we use the modality-specific k.
        builder = PartialObservationGraphBuilder(
            k_neighbors=k,
            min_overlap=min(self.min_features_observed, mod_dim),
            similarity_metric="cosine",
            use_fast_path=True,
            fast_path_threshold=300,
        )

        # Compute similarity matrix among active patients
        sim_matrix = builder.compute_pairwise_similarity(active_features, active_mask)

        # Build kNN graph in local indices
        local_edge_index, edge_weight = builder.build_knn_graph(sim_matrix)

        # Map local indices back to global patient indices
        if local_edge_index.shape[1] > 0:
            global_src = active[local_edge_index[0].numpy()]
            global_dst = active[local_edge_index[1].numpy()]
            global_edge_index = torch.tensor(
                [global_src.tolist(), global_dst.tolist()],
                dtype=torch.long,
            )
        else:
            global_edge_index = torch.zeros((2, 0), dtype=torch.long)

        num_edges = global_edge_index.shape[1]
        logger.debug(
            "Modality '%s': %d active patients, %d edges.",
            mod_name,
            len(active),
            num_edges,
        )

        return {
            "edge_index": global_edge_index,
            "edge_weight": edge_weight,
            "active_patients": active,
            "num_active_patients": len(active),
            "modality_name": mod_name,
            "modality_dim": mod_dim,
            "similarity_matrix": sim_matrix,
        }

    def build_all_subgraphs(
        self,
        features: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        """Build kNN subgraphs for all modalities.

        This is the main entry point. For each modality, extracts the
        relevant features and mask, identifies active patients, computes
        pairwise similarity, and builds a kNN graph.

        Args:
            features: Full feature matrix of shape (N, D_total) where
                ``D_total = sum(modality_dims)``.
            mask: Binary observation mask of shape (N, D_total).

        Returns:
            Dictionary mapping modality name to its subgraph dictionary.
            Each subgraph dict contains the keys documented in
            :meth:`build_subgraph`.

        Raises:
            ValueError: If feature dimensions do not match expectations.
        """
        N = features.shape[0]
        D = features.shape[1]

        if D != self._total_features:
            raise ValueError(
                f"Feature dimension {D} does not match sum of "
                f"modality_dims ({self._total_features}). "
                f"modality_dims={self.modality_dims}"
            )
        if mask.shape != features.shape:
            raise ValueError(
                f"mask shape {mask.shape} != features shape {features.shape}"
            )

        logger.info(
            "Building %d modality subgraphs for %d patients.",
            self.num_modalities,
            N,
        )

        subgraphs: dict[str, dict[str, Any]] = {}

        for m_idx in range(self.num_modalities):
            name = self.modality_names[m_idx]
            subgraphs[name] = self.build_subgraph(features, mask, m_idx)

        # Log summary statistics
        total_edges = sum(sg["edge_index"].shape[1] for sg in subgraphs.values())
        active_counts = [sg["num_active_patients"] for sg in subgraphs.values()]
        logger.info(
            "Modality subgraphs built: %d total edges across %d modalities. "
            "Active patients per modality: min=%d, max=%d, mean=%.1f.",
            total_edges,
            self.num_modalities,
            min(active_counts) if active_counts else 0,
            max(active_counts) if active_counts else 0,
            np.mean(active_counts) if active_counts else 0.0,
        )

        return subgraphs

    def merge_subgraphs(
        self,
        subgraphs: dict[str, dict[str, Any]],
        merge_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge per-modality subgraphs into a single multi-relational edge set.

        This produces a combined edge list with per-modality edge type
        labels, suitable for relational graph neural networks. Edges from
        different modalities between the same pair of patients are kept
        as separate edges with different relation types.

        Args:
            subgraphs: Dictionary mapping modality name to subgraph dict,
                as returned by :meth:`build_all_subgraphs`.
            merge_weights: Optional dictionary mapping modality name to
                a scalar weight applied to that modality's edge weights.
                If None, all modalities are weighted equally (weight=1.0).

        Returns:
            Tuple of:
                - ``edge_index``: (2, E_total) long tensor with global
                  patient indices.
                - ``edge_weight``: (E_total,) float tensor of weighted
                  edge similarities.
                - ``edge_type``: (E_total,) long tensor with modality
                  indices (0 to num_modalities-1).
        """
        all_src = []
        all_dst = []
        all_weight = []
        all_type = []

        for m_idx, name in enumerate(self.modality_names):
            if name not in subgraphs:
                continue

            sg = subgraphs[name]
            ei = sg["edge_index"]
            ew = sg["edge_weight"]

            if ei.shape[1] == 0:
                continue

            weight_scale = 1.0
            if merge_weights is not None and name in merge_weights:
                weight_scale = merge_weights[name]

            all_src.append(ei[0])
            all_dst.append(ei[1])
            all_weight.append(ew * weight_scale)
            all_type.append(torch.full((ei.shape[1],), m_idx, dtype=torch.long))

        if not all_src:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        edge_index = torch.stack(
            [
                torch.cat(all_src),
                torch.cat(all_dst),
            ]
        )
        edge_weight = torch.cat(all_weight)
        edge_type = torch.cat(all_type)

        return edge_index, edge_weight, edge_type

    def get_patient_modality_availability(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Compute per-patient, per-modality availability indicator.

        Returns a binary matrix indicating which patients have sufficient
        observations in each modality (according to
        ``min_features_observed``).

        Args:
            mask: Full observation mask of shape (N, D_total).

        Returns:
            Binary array of shape (N, num_modalities) where entry (i, m)
            is 1 if patient i has >= min_features_observed observations
            in modality m.
        """
        N = mask.shape[0]
        availability = np.zeros((N, self.num_modalities), dtype=np.float32)

        for m_idx in range(self.num_modalities):
            _, mod_mask = self._get_modality_slice(np.zeros_like(mask), mask, m_idx)
            num_observed = mod_mask.sum(axis=1)
            availability[:, m_idx] = (
                num_observed >= self.min_features_observed
            ).astype(np.float32)

        return availability
