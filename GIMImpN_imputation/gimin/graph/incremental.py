"""Incremental graph extension for online patient addition.

In a clinical deployment scenario, new patients arrive continuously
and the graph must be updated without rebuilding from scratch every
time. This module provides an IncrementalGraphManager that:

    1. Computes similarity between a new patient and all existing patients.
    2. Finds the k nearest neighbors for the new patient.
    3. Adds the corresponding edges to the graph.
    4. Tracks additions and triggers a full rebuild after a configurable
       number of incremental updates (to prevent graph quality
       degradation over time).

The manager also supports extracting local subgraphs around a patient
for efficient inference (computing imputation for one patient does not
require message passing over the entire graph).

Classes:
    IncrementalGraphManager: Manage online addition of new patients.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from gimin.graph.partial_similarity import PartialObservationGraphBuilder

logger = logging.getLogger(__name__)


class IncrementalGraphManager:
    """Manage online addition of new patients to the similarity graph.

    When a new patient arrives:
        1. Compute similarity to all existing patients using the fitted
           scaler from the original graph build.
        2. Find the k nearest neighbors among existing patients.
        3. Add bidirectional edges with appropriate weights.
        4. Track the number of incremental additions since the last full
           rebuild.
        5. Optionally trigger a full graph rebuild when the number of
           additions exceeds ``rebuild_interval``.

    The manager maintains the current state of the graph (features,
    mask, edge_index, edge_weight, overlap_frac) and provides utilities
    for extracting local neighborhoods for efficient inference.

    Args:
        graph_builder: A fitted :class:`PartialObservationGraphBuilder`
            instance used for similarity computation and full rebuilds.
        k_neighbors: Number of nearest neighbors for incremental
            additions. Default: 15.
        rebuild_interval: Number of incremental additions before
            triggering a full graph rebuild. Set to 0 to disable
            automatic rebuilds. Default: 100.
        local_hop_count: Number of hops for extracting local subgraphs
            around a patient. Default: 2.

    Attributes:
        current_features: Feature matrix of shape (N, D) for all patients
            currently in the graph.
        current_mask: Observation mask of shape (N, D).
        current_edge_index: Edge index tensor of shape (2, E).
        current_edge_weight: Edge weight tensor of shape (E,).
        current_overlap_frac: Overlap fraction tensor of shape (E,).
        additions_since_rebuild: Count of incremental additions since the
            last full rebuild.

    Example::

        # Build initial graph
        builder = PartialObservationGraphBuilder(k_neighbors=15)
        result = builder.build_full_graph(features, mask)

        # Create incremental manager
        manager = IncrementalGraphManager(builder, k_neighbors=15)
        manager.initialize(
            features, mask,
            result['edge_index'],
            result['edge_weight'],
            result['overlap_frac'],
        )

        # Add a new patient
        new_info = manager.add_patient(new_features, new_mask)
        print(f"New patient index: {new_info['patient_index']}")
        print(f"Connected to {new_info['num_neighbors']} neighbors")
    """

    def __init__(
        self,
        graph_builder: PartialObservationGraphBuilder,
        k_neighbors: int = 15,
        rebuild_interval: int = 100,
        local_hop_count: int = 2,
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if rebuild_interval < 0:
            raise ValueError(f"rebuild_interval must be >= 0, got {rebuild_interval}")
        if local_hop_count < 1:
            raise ValueError(f"local_hop_count must be >= 1, got {local_hop_count}")

        self.graph_builder = graph_builder
        self.k_neighbors = k_neighbors
        self.rebuild_interval = rebuild_interval
        self.local_hop_count = local_hop_count
        self.additions_since_rebuild: int = 0

        # Current graph state
        self.current_features: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.current_edge_index: torch.Tensor | None = None
        self.current_edge_weight: torch.Tensor | None = None
        self.current_overlap_frac: torch.Tensor | None = None

        # Adjacency list for fast neighbor lookups
        self._adjacency: dict[int, list[tuple[int, float]]] | None = None

        # Track which patients were added incrementally (vs. the original batch)
        self._incremental_patient_indices: set[int] = set()

        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        overlap_frac: torch.Tensor,
    ) -> None:
        """Set the initial graph state from a full graph build.

        Args:
            features: Feature matrix of shape (N, D).
            mask: Observation mask of shape (N, D).
            edge_index: Edge index tensor of shape (2, E).
            edge_weight: Edge weight tensor of shape (E,).
            overlap_frac: Overlap fraction tensor of shape (E,).

        Raises:
            ValueError: If input shapes are inconsistent.
        """
        if features.shape != mask.shape:
            raise ValueError(
                f"features shape {features.shape} != mask shape {mask.shape}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be shape (2, E), got {edge_index.shape}")
        E = edge_index.shape[1]
        if edge_weight.shape != (E,):
            raise ValueError(f"edge_weight shape {edge_weight.shape} != ({E},)")
        if overlap_frac.shape != (E,):
            raise ValueError(f"overlap_frac shape {overlap_frac.shape} != ({E},)")

        self.current_features = features.copy()
        self.current_mask = mask.copy()
        self.current_edge_index = edge_index.clone()
        self.current_edge_weight = edge_weight.clone()
        self.current_overlap_frac = overlap_frac.clone()
        self.additions_since_rebuild = 0
        self._incremental_patient_indices = set()

        # Build adjacency list
        self._build_adjacency()
        self._initialized = True

        logger.info(
            "IncrementalGraphManager initialized: %d patients, %d edges.",
            features.shape[0],
            E,
        )

    def _check_initialized(self) -> None:
        """Raise RuntimeError if the manager has not been initialized."""
        if not self._initialized:
            raise RuntimeError(
                "IncrementalGraphManager not initialized. Call initialize() first."
            )

    # ------------------------------------------------------------------
    # Adjacency list management
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> None:
        """Build adjacency list from the current edge_index and edge_weight."""
        N = self.current_features.shape[0]
        self._adjacency = {i: [] for i in range(N)}

        if self.current_edge_index.shape[1] == 0:
            return

        src = self.current_edge_index[0].numpy()
        dst = self.current_edge_index[1].numpy()
        weights = self.current_edge_weight.numpy()

        for s, d, w in zip(src, dst, weights):
            self._adjacency[int(s)].append((int(d), float(w)))

    def _add_to_adjacency(self, node_idx: int) -> None:
        """Ensure a node exists in the adjacency list."""
        if node_idx not in self._adjacency:
            self._adjacency[node_idx] = []

    # ------------------------------------------------------------------
    # Patient addition
    # ------------------------------------------------------------------

    def add_patient(
        self,
        new_features: np.ndarray,
        new_mask: np.ndarray,
        modality_labels: np.ndarray | None = None,
        modality_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Add one patient to the graph incrementally.

        Computes similarity between the new patient and all existing
        patients, selects the top-k neighbors, and adds bidirectional
        edges. The patient's features and mask are appended to the
        current state.

        Args:
            new_features: Feature vector for the new patient, shape (D,)
                or (1, D).
            new_mask: Observation mask for the new patient, shape (D,)
                or (1, D).
            modality_labels: Optional per-feature modality indices.
            modality_weights: Optional per-modality weights.

        Returns:
            Dictionary with keys:
                - ``'patient_index'``: int, global index of the new
                  patient in the expanded graph.
                - ``'num_neighbors'``: int, number of neighbors connected.
                - ``'neighbor_indices'``: list of int, global indices of
                  connected neighbors.
                - ``'neighbor_weights'``: list of float, similarity values
                  to each neighbor.
                - ``'neighbor_overlaps'``: list of float, feature overlap
                  fractions with each neighbor.
                - ``'should_rebuild'``: bool, whether a full rebuild is
                  recommended.
        """
        self._check_initialized()

        # Ensure shapes
        new_features = np.atleast_2d(new_features)
        new_mask = np.atleast_2d(new_mask)

        if new_features.shape[0] != 1:
            raise ValueError(
                "add_patient expects a single patient. "
                f"Got {new_features.shape[0]} rows."
            )

        D = self.current_features.shape[1]
        if new_features.shape[1] != D:
            raise ValueError(
                f"New patient has {new_features.shape[1]} features, expected {D}."
            )

        new_features_1d = new_features.ravel()
        new_mask_1d = new_mask.ravel()

        # Check if the new patient has any observations at all
        if new_mask_1d.sum() == 0:
            logger.warning(
                "New patient has no observed features. Adding as isolated node."
            )

        # Compute similarity to all existing patients
        similarities = self.graph_builder.compute_similarity_for_new_patient(
            new_features_1d,
            new_mask_1d,
            self.current_features,
            self.current_mask,
            modality_labels=modality_labels,
            modality_weights=modality_weights,
        )

        # Assign the new patient an index
        new_idx = self.current_features.shape[0]

        # Find top-k neighbors
        num_positive = (similarities > 0).sum()
        k = min(self.k_neighbors, num_positive)

        if k == 0:
            neighbor_indices = []
            neighbor_weights = []
            neighbor_overlaps = []
            logger.warning(
                "New patient %d has no positive similarities. Adding as isolated node.",
                new_idx,
            )
        else:
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            # Filter to only positive
            top_k_idx = top_k_idx[similarities[top_k_idx] > 0]
            # Sort by similarity descending for consistent ordering
            top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]

            neighbor_indices = top_k_idx.tolist()
            neighbor_weights = similarities[top_k_idx].tolist()

            # Compute overlap fractions for new edges
            neighbor_overlaps = []
            for j in neighbor_indices:
                overlap = float((new_mask_1d * self.current_mask[j]).sum() / D)
                neighbor_overlaps.append(overlap)

        # Append the new patient's data
        self.current_features = np.vstack([self.current_features, new_features])
        self.current_mask = np.vstack([self.current_mask, new_mask])

        # Add edges (bidirectional)
        if neighbor_indices:
            new_src = []
            new_dst = []
            new_weights = []
            new_overlaps = []

            for j, w, o in zip(neighbor_indices, neighbor_weights, neighbor_overlaps):
                # new_idx -> j
                new_src.append(new_idx)
                new_dst.append(j)
                new_weights.append(w)
                new_overlaps.append(o)
                # j -> new_idx
                new_src.append(j)
                new_dst.append(new_idx)
                new_weights.append(w)
                new_overlaps.append(o)

            new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
            new_edge_weight = torch.tensor(new_weights, dtype=torch.float32)
            new_overlap_frac = torch.tensor(new_overlaps, dtype=torch.float32)

            # Concatenate with existing edges
            self.current_edge_index = torch.cat(
                [self.current_edge_index, new_edge_index], dim=1
            )
            self.current_edge_weight = torch.cat(
                [self.current_edge_weight, new_edge_weight]
            )
            self.current_overlap_frac = torch.cat(
                [self.current_overlap_frac, new_overlap_frac]
            )

        # Update adjacency list
        self._add_to_adjacency(new_idx)
        for j, w in zip(neighbor_indices, neighbor_weights):
            self._adjacency[new_idx].append((j, w))
            self._adjacency[j].append((new_idx, w))

        # Track incremental additions
        self._incremental_patient_indices.add(new_idx)
        self.additions_since_rebuild += 1

        should_rebuild = self.should_rebuild()
        if should_rebuild:
            logger.info(
                "Rebuild recommended: %d additions since last rebuild (threshold=%d).",
                self.additions_since_rebuild,
                self.rebuild_interval,
            )

        logger.debug(
            "Added patient %d with %d neighbors. Total: %d patients, %d edges.",
            new_idx,
            len(neighbor_indices),
            self.current_features.shape[0],
            self.current_edge_index.shape[1],
        )

        return {
            "patient_index": new_idx,
            "num_neighbors": len(neighbor_indices),
            "neighbor_indices": neighbor_indices,
            "neighbor_weights": neighbor_weights,
            "neighbor_overlaps": neighbor_overlaps,
            "should_rebuild": should_rebuild,
        }

    def add_patients_batch(
        self,
        new_features: np.ndarray,
        new_mask: np.ndarray,
        modality_labels: np.ndarray | None = None,
        modality_weights: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Add multiple patients incrementally.

        Each patient is added sequentially so that later patients can
        potentially connect to earlier patients in the same batch.

        Args:
            new_features: Feature matrix of shape (B, D) for B new patients.
            new_mask: Observation mask of shape (B, D).
            modality_labels: Optional per-feature modality indices.
            modality_weights: Optional per-modality weights.

        Returns:
            List of dictionaries, one per patient, as returned by
            :meth:`add_patient`.
        """
        self._check_initialized()

        new_features = np.atleast_2d(new_features)
        new_mask = np.atleast_2d(new_mask)
        B = new_features.shape[0]

        results = []
        for b in range(B):
            result = self.add_patient(
                new_features[b : b + 1],
                new_mask[b : b + 1],
                modality_labels=modality_labels,
                modality_weights=modality_weights,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Rebuild management
    # ------------------------------------------------------------------

    def should_rebuild(self) -> bool:
        """Check if a full graph rebuild is recommended.

        Returns True if the number of incremental additions since the
        last rebuild exceeds ``rebuild_interval``. If ``rebuild_interval``
        is 0, always returns False (auto-rebuild disabled).

        Returns:
            True if a rebuild is recommended.
        """
        if self.rebuild_interval == 0:
            return False
        return self.additions_since_rebuild >= self.rebuild_interval

    def full_rebuild(
        self,
        modality_labels: np.ndarray | None = None,
        modality_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Rebuild the entire graph from scratch using current features.

        This should be called periodically to maintain graph quality,
        as incremental additions can lead to suboptimal neighborhoods
        over time (new patients only connect to the existing graph,
        not to each other unless added sequentially).

        Args:
            modality_labels: Optional per-feature modality indices.
            modality_weights: Optional per-modality weights.

        Returns:
            Dictionary with the full graph build results (same keys as
            ``PartialObservationGraphBuilder.build_full_graph``), plus
            ``'patients_before_rebuild'`` and ``'additions_rebuilt'``.
        """
        self._check_initialized()

        N = self.current_features.shape[0]
        num_additions = self.additions_since_rebuild

        logger.info(
            "Performing full graph rebuild: %d patients (%d added since last rebuild).",
            N,
            num_additions,
        )

        # Full rebuild
        result = self.graph_builder.build_full_graph(
            self.current_features,
            self.current_mask,
            modality_labels=modality_labels,
            modality_weights=modality_weights,
        )

        # Update state
        self.current_edge_index = result["edge_index"]
        self.current_edge_weight = result["edge_weight"]
        self.current_overlap_frac = result["overlap_frac"]
        self.additions_since_rebuild = 0
        self._incremental_patient_indices.clear()

        # Rebuild adjacency
        self._build_adjacency()

        result["patients_before_rebuild"] = N
        result["additions_rebuilt"] = num_additions

        return result

    # ------------------------------------------------------------------
    # Local subgraph extraction
    # ------------------------------------------------------------------

    def extract_local_subgraph(
        self,
        center_node: int,
        num_hops: int | None = None,
    ) -> dict[str, Any]:
        """Extract a local subgraph around a given patient.

        Performs a BFS from ``center_node`` up to ``num_hops`` hops to
        collect a local neighborhood. Returns the subgraph's edge_index
        (in local indices), a mapping from local to global indices, and
        the relevant features and mask.

        This is useful for efficient inference: when computing an
        imputation for a single patient, only the local neighborhood
        needs to participate in message passing.

        Args:
            center_node: Global index of the center patient.
            num_hops: Number of hops to expand. Default: uses
                ``self.local_hop_count``.

        Returns:
            Dictionary with keys:
                - ``'local_edge_index'``: (2, E_local) long tensor with
                  local node indices.
                - ``'local_edge_weight'``: (E_local,) float tensor.
                - ``'local_overlap_frac'``: (E_local,) float tensor.
                - ``'local_features'``: (N_local, D) numpy array.
                - ``'local_mask'``: (N_local, D) numpy array.
                - ``'global_to_local'``: dict mapping global -> local idx.
                - ``'local_to_global'``: dict mapping local -> global idx.
                - ``'center_local_idx'``: int, local index of center_node.
                - ``'num_local_nodes'``: int.

        Raises:
            ValueError: If center_node is out of bounds.
        """
        self._check_initialized()

        N = self.current_features.shape[0]
        if center_node < 0 or center_node >= N:
            raise ValueError(f"center_node {center_node} out of bounds [0, {N}).")

        if num_hops is None:
            num_hops = self.local_hop_count

        # BFS to collect nodes within num_hops
        visited: set[int] = {center_node}
        frontier: set[int] = {center_node}

        for _ in range(num_hops):
            next_frontier: set[int] = set()
            for node in frontier:
                if node in self._adjacency:
                    for neighbor, _ in self._adjacency[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break  # No more nodes to explore

        # Create local-to-global and global-to-local mappings
        # Sort for deterministic ordering; center node gets index 0
        local_nodes = sorted(visited - {center_node})
        local_nodes = [center_node] + local_nodes

        global_to_local = {g: l for l, g in enumerate(local_nodes)}
        local_to_global = {l: g for l, g in enumerate(local_nodes)}

        # Extract features and mask for local nodes
        local_indices = np.array(local_nodes)
        local_features = self.current_features[local_indices]
        local_mask = self.current_mask[local_indices]

        # Extract edges that are entirely within the local subgraph
        local_src = []
        local_dst = []
        local_weights = []
        local_overlaps = []

        if self.current_edge_index.shape[1] > 0:
            all_src = self.current_edge_index[0].numpy()
            all_dst = self.current_edge_index[1].numpy()
            all_w = self.current_edge_weight.numpy()
            all_o = self.current_overlap_frac.numpy()

            for idx in range(len(all_src)):
                s, d = int(all_src[idx]), int(all_dst[idx])
                if s in global_to_local and d in global_to_local:
                    local_src.append(global_to_local[s])
                    local_dst.append(global_to_local[d])
                    local_weights.append(float(all_w[idx]))
                    local_overlaps.append(float(all_o[idx]))

        if local_src:
            local_edge_index = torch.tensor([local_src, local_dst], dtype=torch.long)
            local_edge_weight = torch.tensor(local_weights, dtype=torch.float32)
            local_overlap_frac = torch.tensor(local_overlaps, dtype=torch.float32)
        else:
            local_edge_index = torch.zeros((2, 0), dtype=torch.long)
            local_edge_weight = torch.zeros(0, dtype=torch.float32)
            local_overlap_frac = torch.zeros(0, dtype=torch.float32)

        return {
            "local_edge_index": local_edge_index,
            "local_edge_weight": local_edge_weight,
            "local_overlap_frac": local_overlap_frac,
            "local_features": local_features,
            "local_mask": local_mask,
            "global_to_local": global_to_local,
            "local_to_global": local_to_global,
            "center_local_idx": 0,  # center_node is always index 0
            "num_local_nodes": len(local_nodes),
        }

    # ------------------------------------------------------------------
    # Queries and diagnostics
    # ------------------------------------------------------------------

    @property
    def num_patients(self) -> int:
        """Current number of patients in the graph."""
        if self.current_features is None:
            return 0
        return self.current_features.shape[0]

    @property
    def num_edges(self) -> int:
        """Current number of directed edges in the graph."""
        if self.current_edge_index is None:
            return 0
        return self.current_edge_index.shape[1]

    @property
    def num_incremental_patients(self) -> int:
        """Number of patients added incrementally since last rebuild."""
        return len(self._incremental_patient_indices)

    def get_neighbors(self, patient_idx: int) -> list[tuple[int, float]]:
        """Get the neighbors and edge weights for a given patient.

        Args:
            patient_idx: Global index of the patient.

        Returns:
            List of (neighbor_index, edge_weight) tuples, sorted by
            weight descending.

        Raises:
            ValueError: If patient_idx is out of bounds.
        """
        self._check_initialized()

        if patient_idx < 0 or patient_idx >= self.num_patients:
            raise ValueError(
                f"patient_idx {patient_idx} out of bounds [0, {self.num_patients})."
            )

        neighbors = self._adjacency.get(patient_idx, [])
        return sorted(neighbors, key=lambda x: -x[1])

    def get_degree(self, patient_idx: int) -> int:
        """Get the degree (number of neighbors) for a given patient.

        Args:
            patient_idx: Global index of the patient.

        Returns:
            Number of neighbors.
        """
        self._check_initialized()
        return len(self._adjacency.get(patient_idx, []))

    def get_isolated_patients(self) -> list[int]:
        """Get indices of patients with no graph neighbors.

        Returns:
            List of global patient indices with degree 0.
        """
        self._check_initialized()
        return [
            i for i in range(self.num_patients) if len(self._adjacency.get(i, [])) == 0
        ]

    def get_graph_statistics(self) -> dict[str, Any]:
        """Compute summary statistics about the current graph.

        Returns:
            Dictionary with keys:
                - ``'num_patients'``: int
                - ``'num_edges'``: int (directed)
                - ``'num_isolated'``: int
                - ``'num_incremental'``: int (patients added incrementally)
                - ``'additions_since_rebuild'``: int
                - ``'mean_degree'``: float
                - ``'median_degree'``: float
                - ``'max_degree'``: int
                - ``'min_degree'``: int (over non-isolated nodes)
                - ``'mean_edge_weight'``: float
                - ``'mean_overlap_frac'``: float
        """
        self._check_initialized()

        degrees = [len(self._adjacency.get(i, [])) for i in range(self.num_patients)]
        non_isolated_degrees = [d for d in degrees if d > 0]

        stats: dict[str, Any] = {
            "num_patients": self.num_patients,
            "num_edges": self.num_edges,
            "num_isolated": sum(1 for d in degrees if d == 0),
            "num_incremental": self.num_incremental_patients,
            "additions_since_rebuild": self.additions_since_rebuild,
            "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            "median_degree": float(np.median(degrees)) if degrees else 0.0,
            "max_degree": max(degrees) if degrees else 0,
        }

        if non_isolated_degrees:
            stats["min_degree"] = min(non_isolated_degrees)
        else:
            stats["min_degree"] = 0

        if self.num_edges > 0:
            stats["mean_edge_weight"] = float(self.current_edge_weight.mean())
            stats["mean_overlap_frac"] = float(self.current_overlap_frac.mean())
        else:
            stats["mean_edge_weight"] = 0.0
            stats["mean_overlap_frac"] = 0.0

        return stats

    def get_current_graph(self) -> dict[str, Any]:
        """Return the current graph state as a dictionary.

        This is the same format as returned by
        ``PartialObservationGraphBuilder.build_full_graph`` (minus the
        similarity_matrix, which is not maintained incrementally).

        Returns:
            Dictionary with keys edge_index, edge_weight, overlap_frac,
            num_nodes, num_edges.
        """
        self._check_initialized()

        return {
            "edge_index": self.current_edge_index,
            "edge_weight": self.current_edge_weight,
            "overlap_frac": self.current_overlap_frac,
            "num_nodes": self.num_patients,
            "num_edges": self.num_edges,
        }

    def remove_patient(self, patient_idx: int) -> None:
        """Remove a patient from the graph.

        This removes the patient's features, mask, and all associated
        edges. Remaining patient indices are renumbered (shifted down)
        to fill the gap.

        Note: This is an O(N + E) operation due to index remapping.
        For frequent removals, consider batching them and performing
        a full rebuild instead.

        Args:
            patient_idx: Global index of the patient to remove.

        Raises:
            ValueError: If patient_idx is out of bounds.
        """
        self._check_initialized()

        N = self.num_patients
        if patient_idx < 0 or patient_idx >= N:
            raise ValueError(f"patient_idx {patient_idx} out of bounds [0, {N}).")

        logger.info("Removing patient %d from graph (%d patients).", patient_idx, N)

        # Remove from features and mask
        self.current_features = np.delete(self.current_features, patient_idx, axis=0)
        self.current_mask = np.delete(self.current_mask, patient_idx, axis=0)

        # Remove edges involving this patient and remap indices
        if self.current_edge_index.shape[1] > 0:
            src = self.current_edge_index[0].numpy()
            dst = self.current_edge_index[1].numpy()
            weights = self.current_edge_weight.numpy()
            overlaps = self.current_overlap_frac.numpy()

            # Keep edges that don't involve patient_idx
            keep = (src != patient_idx) & (dst != patient_idx)

            src = src[keep]
            dst = dst[keep]
            weights = weights[keep]
            overlaps = overlaps[keep]

            # Remap indices: shift down indices > patient_idx
            src = np.where(src > patient_idx, src - 1, src)
            dst = np.where(dst > patient_idx, dst - 1, dst)

            self.current_edge_index = torch.tensor(
                np.stack([src, dst]), dtype=torch.long
            )
            self.current_edge_weight = torch.tensor(weights, dtype=torch.float32)
            self.current_overlap_frac = torch.tensor(overlaps, dtype=torch.float32)
        else:
            # No edges to modify
            pass

        # Update incremental tracking
        new_incremental = set()
        for idx in self._incremental_patient_indices:
            if idx == patient_idx:
                continue
            elif idx > patient_idx:
                new_incremental.add(idx - 1)
            else:
                new_incremental.add(idx)
        self._incremental_patient_indices = new_incremental

        # Rebuild adjacency
        self._build_adjacency()

        logger.info(
            "Patient removed. Now %d patients, %d edges.",
            self.num_patients,
            self.num_edges,
        )
