"""Incremental (online) imputation for new patients without retraining.

When new patients arrive, the full GIMIN model need not be retrained.
Instead, :class:`IncrementalGIMIN`:

1. Finds the *k* nearest neighbours of the new patient in the existing
   graph using the available features.
2. Runs local message passing over the new patient's neighbourhood.
3. Returns imputed values and uncertainty estimates.

If enough new patients accumulate (controlled by ``rebuild_interval``),
the graph can be rebuilt to incorporate the new data.

Typical usage::

    from gimin.inference.incremental import IncrementalGIMIN

    inc = IncrementalGIMIN(model, graph_state, config)

    for patient_features, patient_mask in new_patient_stream:
        result = inc.add_patient(patient_features, patient_mask)
        print(result["imputed"], result["pred_std"])

    if inc.should_rebuild_graph():
        inc.rebuild_graph()
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..config import GIMINConfig
from ..utils import ArrayLike
from ..utils import to_tensor as _to_tensor

logger = logging.getLogger(__name__)


class IncrementalGIMIN:
    """Online imputation for new patients without retraining.

    Maintains the frozen GIMIN model weights and a mutable graph state.
    New patients are connected to existing patients via kNN, and local
    message passing is performed to produce imputed values.

    Args:
        model: A trained :class:`~gimin.model.gimin_core.GIMIN` instance
            with frozen weights.
        graph_state: Dictionary containing the current graph and patient
            data.  Expected keys:

            - ``"features"``: existing patient feature matrix ``(N, F)``
            - ``"mask"``: existing observation mask ``(N, F)``
            - ``"edge_index"``: graph edge indices ``(2, E)``
            - ``"edge_weight"``: edge weights ``(E,)``

        config: GIMIN configuration.
        device: Torch device.  Default: auto-detect.
    """

    def __init__(
        self,
        model: nn.Module,
        graph_state: dict[str, torch.Tensor],
        config: GIMINConfig,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.model = model.to(self.device)
        self.model.eval()

        # Freeze model parameters.
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialise graph state.
        self._features = _to_tensor(graph_state["features"], self.device)
        self._mask = _to_tensor(graph_state["mask"], self.device)
        self._edge_index = graph_state["edge_index"].to(self.device)
        self._edge_weight = graph_state.get("edge_weight")
        if self._edge_weight is not None:
            self._edge_weight = self._edge_weight.to(self.device)
        else:
            self._edge_weight = torch.ones(
                self._edge_index.shape[1], device=self.device
            )

        # Tracking for incremental additions.
        self._num_original_patients: int = self._features.shape[0]
        self._num_added_patients: int = 0
        self._rebuild_interval: int = config.incremental.rebuild_interval
        self._k_neighbors: int = config.graph.k_neighbors
        self._local_hops: int = config.incremental.local_hop_count

        # Optional: graph builder for full rebuilds.
        self._graph_builder: Any | None = None

    # ------------------------------------------------------------------
    # Graph builder registration
    # ------------------------------------------------------------------

    def set_graph_builder(self, graph_builder: Any) -> None:
        """Register a graph builder for full graph rebuilds.

        Args:
            graph_builder: Object with a ``build(features, mask)`` method
                returning ``(edge_index, edge_weight)``.
        """
        self._graph_builder = graph_builder

    # ------------------------------------------------------------------
    # kNN edge computation for new patients
    # ------------------------------------------------------------------

    def _compute_knn_edges(
        self,
        new_features: torch.Tensor,
        new_mask: torch.Tensor,
        k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute kNN edges connecting a new patient to existing patients.

        Uses cosine similarity over the shared observed features between
        the new patient and each existing patient.

        Args:
            new_features: Feature vector for the new patient, shape ``(1, F)``
                or ``(F,)``.
            new_mask: Observation mask for the new patient, same shape.
            k: Number of neighbours.  Default: ``self._k_neighbors``.

        Returns:
            Tuple of ``(edge_index, edge_weight)`` for the new edges.
            ``edge_index`` has shape ``(2, 2*k)`` (bidirectional) and
            ``edge_weight`` has shape ``(2*k,)``.
        """
        k = k or self._k_neighbors

        if new_features.dim() == 1:
            new_features = new_features.unsqueeze(0)
        if new_mask.dim() == 1:
            new_mask = new_mask.unsqueeze(0)

        new_idx = self._features.shape[0]  # Index for the new patient.
        existing_features = self._features  # (N_existing, F)
        existing_mask = self._mask

        # Compute shared-feature similarity.
        # For each existing patient, find features observed in both.
        new_feat = new_features[0]  # (F,)
        new_m = new_mask[0].bool()  # (F,)

        similarities = []
        for i in range(existing_features.shape[0]):
            shared = new_m & existing_mask[i].bool()
            num_shared = shared.sum().item()

            if num_shared < self.config.graph.min_overlap:
                similarities.append(-1.0)
                continue

            v_new = new_feat[shared]
            v_existing = existing_features[i][shared]

            # Cosine similarity.
            norm_new = torch.norm(v_new) + 1e-8
            norm_existing = torch.norm(v_existing) + 1e-8
            cos_sim = (v_new @ v_existing) / (norm_new * norm_existing)
            similarities.append(cos_sim.item())

        sim_array = torch.tensor(similarities, device=self.device)

        # Select top-k neighbours (excluding negative/insufficient overlap).
        valid_mask = sim_array > -0.5
        if valid_mask.sum() < k:
            k = max(1, int(valid_mask.sum().item()))
            logger.warning(
                "New patient has fewer than %d valid neighbours; using k=%d.",
                self._k_neighbors,
                k,
            )

        # Set invalid similarities to -inf for top-k selection.
        sim_array[~valid_mask] = float("-inf")
        topk_values, topk_indices = torch.topk(sim_array, k)

        # Build bidirectional edges.
        src_to_new = topk_indices.long()
        dst_to_new = torch.full((k,), new_idx, dtype=torch.long, device=self.device)

        edge_index = torch.stack(
            [
                torch.cat([src_to_new, dst_to_new]),
                torch.cat([dst_to_new, src_to_new]),
            ],
            dim=0,
        )

        # Normalise similarities to [0, 1] for edge weights.
        edge_weights_half = torch.clamp(topk_values, min=0.0)
        edge_weight = torch.cat([edge_weights_half, edge_weights_half])

        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # Local neighbourhood extraction
    # ------------------------------------------------------------------

    def _get_local_subgraph(
        self,
        center_idx: int,
        num_hops: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Extract a local subgraph around a node.

        Args:
            center_idx: Index of the center node.
            num_hops: Number of hops to expand.

        Returns:
            Tuple of ``(local_edge_index, local_edge_weight, node_indices)``
            where ``node_indices`` maps local indices to global indices.
        """
        edge_src = self._edge_index[0]
        edge_dst = self._edge_index[1]

        visited = {center_idx}
        frontier = {center_idx}

        for _ in range(num_hops):
            new_frontier = set()
            for node in frontier:
                # Find all neighbours of this node.
                neighbor_mask = edge_src == node
                neighbours = edge_dst[neighbor_mask].tolist()
                for n in neighbours:
                    if n not in visited:
                        visited.add(n)
                        new_frontier.add(n)
                # Also check reverse direction.
                neighbor_mask_rev = edge_dst == node
                neighbours_rev = edge_src[neighbor_mask_rev].tolist()
                for n in neighbours_rev:
                    if n not in visited:
                        visited.add(n)
                        new_frontier.add(n)
            frontier = new_frontier

        node_list = sorted(visited)
        node_set = set(node_list)
        global_to_local = {g: l for l, g in enumerate(node_list)}

        # Extract edges within the local subgraph.
        local_src = []
        local_dst = []
        local_weights = []

        for e in range(self._edge_index.shape[1]):
            s = edge_src[e].item()
            d = edge_dst[e].item()
            if s in node_set and d in node_set:
                local_src.append(global_to_local[s])
                local_dst.append(global_to_local[d])
                if self._edge_weight is not None:
                    local_weights.append(self._edge_weight[e].item())
                else:
                    local_weights.append(1.0)

        local_edge_index = torch.tensor(
            [local_src, local_dst], dtype=torch.long, device=self.device
        )
        local_edge_weight = torch.tensor(
            local_weights, dtype=torch.float32, device=self.device
        )

        return local_edge_index, local_edge_weight, node_list

    # ------------------------------------------------------------------
    # Single patient addition
    # ------------------------------------------------------------------

    def add_patient(
        self,
        new_features: ArrayLike,
        new_mask: ArrayLike,
        mc_samples: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Add one patient: compute kNN edges, run local message passing.

        The new patient is appended to the internal feature and mask
        matrices, connected to the graph via kNN edges, and then the
        model runs inference on the local neighbourhood.

        Args:
            new_features: Feature vector, shape ``(F,)`` or ``(1, F)``.
            new_mask: Observation mask, same shape as *new_features*.
            mc_samples: Number of MC dropout samples for uncertainty.
                Default: ``config.evaluation.mc_samples``.

        Returns:
            Dictionary with:

            - ``"imputed"``: imputed feature vector, shape ``(F,)``
            - ``"pred_std"``: uncertainty estimate, shape ``(F,)``
            - ``"patient_index"``: integer index of the new patient in
              the graph
        """
        mc_samples = mc_samples or self.config.evaluation.mc_samples
        new_feat = _to_tensor(new_features, self.device)
        new_m = _to_tensor(new_mask, self.device)

        if new_feat.dim() == 1:
            new_feat = new_feat.unsqueeze(0)
        if new_m.dim() == 1:
            new_m = new_m.unsqueeze(0)

        # Zero out missing values.
        new_feat = new_feat * new_m

        # Append to internal state.
        new_idx = self._features.shape[0]
        self._features = torch.cat([self._features, new_feat], dim=0)
        self._mask = torch.cat([self._mask, new_m], dim=0)

        # Compute kNN edges for the new patient.
        new_edges, new_weights = self._compute_knn_edges(new_feat, new_m)
        self._edge_index = torch.cat([self._edge_index, new_edges], dim=1)
        self._edge_weight = torch.cat([self._edge_weight, new_weights])

        self._num_added_patients += 1

        # Extract local subgraph and run inference.
        local_ei, local_ew, node_list = self._get_local_subgraph(
            new_idx, self._local_hops
        )

        local_features = self._features[node_list]  # (L, F)
        local_mask = self._mask[node_list]  # (L, F)

        # Find the local index of the new patient.
        local_new_idx = node_list.index(new_idx)

        # MC dropout inference on local subgraph.
        mc_predictions: list[np.ndarray] = []

        for s in range(mc_samples):
            if s == 0:
                self.model.eval()  # First pass: deterministic.
            else:
                self.model.train()  # Subsequent passes: with dropout.

            with torch.no_grad():
                output = self.model(
                    features=local_features,
                    mask=local_mask,
                    edge_index=local_ei,
                    edge_weight=local_ew,
                    overlap_frac=torch.ones(local_ei.shape[1], device=self.device),
                    modality_dims=self.config.modality_dims,
                )
                mc_predictions.append(output["imputed"][local_new_idx].cpu().numpy())

        self.model.eval()

        mc_stack = np.stack(mc_predictions, axis=0)  # (S, F)
        pred_mean = mc_stack.mean(axis=0)  # (F,)
        pred_std = mc_stack.std(axis=0)  # (F,)

        # Update internal features with imputed values (for future kNN).
        imputed_tensor = torch.from_numpy(pred_mean.astype(np.float32)).to(self.device)
        self._features[new_idx] = (
            new_m[0] * self._features[new_idx] + (1.0 - new_m[0]) * imputed_tensor
        )

        return {
            "imputed": pred_mean,
            "pred_std": pred_std,
            "patient_index": new_idx,
        }

    # ------------------------------------------------------------------
    # Batch patient addition
    # ------------------------------------------------------------------

    def batch_add_patients(
        self,
        new_features: ArrayLike,
        new_masks: ArrayLike,
        mc_samples: int | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Efficiently add a batch of new patients.

        Processes each patient sequentially so that earlier patients
        can serve as neighbours for later ones.

        Args:
            new_features: Feature matrix for new patients, shape ``(B, F)``.
            new_masks: Observation masks, shape ``(B, F)``.
            mc_samples: MC dropout samples per patient.

        Returns:
            List of result dictionaries (one per patient), each with
            keys ``"imputed"``, ``"pred_std"``, ``"patient_index"``.
        """
        if isinstance(new_features, np.ndarray):
            new_features_np = new_features
        else:
            new_features_np = new_features.detach().cpu().numpy()

        if isinstance(new_masks, np.ndarray):
            new_masks_np = new_masks
        else:
            new_masks_np = new_masks.detach().cpu().numpy()

        batch_size = new_features_np.shape[0]
        results: list[dict[str, np.ndarray]] = []

        logger.info("Adding batch of %d new patients.", batch_size)

        for i in range(batch_size):
            result = self.add_patient(
                new_features_np[i],
                new_masks_np[i],
                mc_samples=mc_samples,
            )
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info("  Processed %d/%d new patients.", i + 1, batch_size)

        return results

    # ------------------------------------------------------------------
    # Graph rebuild management
    # ------------------------------------------------------------------

    def should_rebuild_graph(self) -> bool:
        """Check whether enough patients have been added to warrant a rebuild.

        Returns:
            ``True`` if ``num_added_patients >= rebuild_interval``.
        """
        return self._num_added_patients >= self._rebuild_interval

    def rebuild_graph(self) -> None:
        """Full graph rebuild incorporating all patients (original + new).

        Uses the registered graph builder to construct a new kNN graph
        from scratch.  Resets the added-patient counter.

        Raises:
            RuntimeError: If no graph builder has been registered via
                :meth:`set_graph_builder`.
        """
        if self._graph_builder is None:
            raise RuntimeError(
                "No graph builder registered.  Call set_graph_builder() "
                "before rebuilding."
            )

        logger.info(
            "Rebuilding graph with %d patients (%d original + %d added).",
            self._features.shape[0],
            self._num_original_patients,
            self._num_added_patients,
        )

        features_cpu = self._features.cpu()
        mask_cpu = self._mask.cpu()

        if hasattr(self._graph_builder, "build"):
            edge_index, edge_weight = self._graph_builder.build(
                features_cpu, mask=mask_cpu
            )
        elif callable(self._graph_builder):
            edge_index, edge_weight = self._graph_builder(features_cpu, mask=mask_cpu)
        else:
            raise TypeError(
                f"graph_builder must be callable or have a 'build' method, "
                f"got {type(self._graph_builder)}"
            )

        self._edge_index = edge_index.to(self.device)
        self._edge_weight = edge_weight.to(self.device)

        # Update counters.
        self._num_original_patients = self._features.shape[0]
        self._num_added_patients = 0

        logger.info("Graph rebuild complete.  New edge count: %d", edge_index.shape[1])

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @property
    def num_patients(self) -> int:
        """Total number of patients in the current graph."""
        return self._features.shape[0]

    @property
    def num_added_patients(self) -> int:
        """Number of patients added since last rebuild."""
        return self._num_added_patients

    def get_graph_state(self) -> dict[str, torch.Tensor]:
        """Return the current graph state for serialization.

        Returns:
            Dictionary with ``"features"``, ``"mask"``, ``"edge_index"``,
            and ``"edge_weight"`` tensors.
        """
        return {
            "features": self._features.cpu(),
            "mask": self._mask.cpu(),
            "edge_index": self._edge_index.cpu(),
            "edge_weight": self._edge_weight.cpu(),
        }

    def save_state(self, path: str) -> None:
        """Save the full incremental state to disk.

        Args:
            path: Output file path.
        """
        from pathlib import Path as P

        filepath = P(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = self.get_graph_state()
        state["num_original_patients"] = self._num_original_patients
        state["num_added_patients"] = self._num_added_patients

        torch.save(state, filepath)
        logger.info("Incremental state saved to %s", filepath)

    def load_state(self, path: str) -> None:
        """Load incremental state from disk.

        Args:
            path: Path to a previously saved state file.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        from pathlib import Path as P

        filepath = P(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Incremental state not found: {filepath}")

        state = torch.load(filepath, map_location=self.device, weights_only=True)

        self._features = state["features"].to(self.device)
        self._mask = state["mask"].to(self.device)
        self._edge_index = state["edge_index"].to(self.device)
        self._edge_weight = state["edge_weight"].to(self.device)
        self._num_original_patients = state.get(
            "num_original_patients", self._features.shape[0]
        )
        self._num_added_patients = state.get("num_added_patients", 0)

        logger.info(
            "Incremental state loaded from %s (%d patients).",
            filepath,
            self._features.shape[0],
        )
