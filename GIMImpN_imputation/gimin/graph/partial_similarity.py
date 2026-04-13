"""Partial-observation patient similarity graph construction for GIMIN.

Solves the chicken-and-egg problem: we need features to build the graph,
but we need the graph to impute features. Solution: compute pairwise
cosine similarity using only mutually observed features.

The core algorithm for a pair of patients (i, j):
    1. Identify shared features: shared = mask[i] & mask[j]
    2. If |shared| < min_overlap, similarity is 0 (insufficient evidence).
    3. Otherwise, compute cosine similarity on standardized shared features.
    4. Weight by sqrt(|shared| / D) to penalize low-overlap pairs.

This module provides both an exact O(N^2 * D) implementation and a
vectorized fast path for datasets with N > 2000 patients.

Classes:
    PartialObservationGraphBuilder: End-to-end graph construction from
        partially observed multimodal features.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PartialObservationGraphBuilder:
    """Build patient similarity graph from partially observed multimodal features.

    This class implements the key innovation of GIMIN: constructing a
    meaningful patient similarity graph even when each patient has a
    different subset of observed features. Pairwise similarity is computed
    using only the features that both patients have observed, with an
    overlap penalty to down-weight pairs with few shared features.

    Args:
        k_neighbors: Number of nearest neighbors per node in the kNN graph.
            The final graph is symmetrized, so actual degree may be up to 2k.
        min_overlap: Minimum number of mutually observed features required
            to compute similarity. Pairs below this threshold get sim = 0.
        similarity_metric: Similarity function to use. Currently only
            ``'cosine'`` is supported.
        use_fast_path: If True, automatically use the vectorized
            ``compute_pairwise_similarity_fast`` method when the number of
            patients exceeds ``fast_path_threshold``.
        fast_path_threshold: Patient count above which the fast path is
            used (default 500). The fast path uses more memory but is
            significantly faster for large datasets.

    Attributes:
        scaler_means_: Per-feature means fitted from observed values only.
            Shape (D,). Set after calling :meth:`fit_scaler`.
        scaler_stds_: Per-feature standard deviations fitted from observed
            values only. Shape (D,). Set after calling :meth:`fit_scaler`.

    Example::

        builder = PartialObservationGraphBuilder(k_neighbors=15, min_overlap=3)
        features = np.random.randn(100, 39)
        mask = (np.random.rand(100, 39) > 0.3).astype(np.float32)
        features[mask == 0] = np.nan

        result = builder.build_full_graph(features, mask)
        print(result['edge_index'].shape)   # (2, E)
        print(result['edge_weight'].shape)  # (E,)
    """

    def __init__(
        self,
        k_neighbors: int = 15,
        min_overlap: int = 3,
        similarity_metric: str = "cosine",
        use_fast_path: bool = True,
        fast_path_threshold: int = 500,
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if min_overlap < 1:
            raise ValueError(f"min_overlap must be >= 1, got {min_overlap}")
        if similarity_metric != "cosine":
            raise ValueError(
                f"Only 'cosine' similarity is supported, got '{similarity_metric}'"
            )

        self.k_neighbors = k_neighbors
        self.min_overlap = min_overlap
        self.similarity_metric = similarity_metric
        self.use_fast_path = use_fast_path
        self.fast_path_threshold = fast_path_threshold

        # Fitted scaling parameters (from observed values only)
        self.scaler_means_: np.ndarray | None = None
        self.scaler_stds_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def fit_scaler(self, features: np.ndarray, mask: np.ndarray) -> None:
        """Compute per-feature mean and std from observed values only.

        For each feature dimension j, the mean and standard deviation are
        computed using only those patients where ``mask[:, j] == 1``.
        Features with fewer than 2 observed values or near-zero variance
        get a standard deviation of 1.0 to avoid division by zero.

        Args:
            features: Feature matrix of shape (N, D). Missing entries may
                contain NaN or arbitrary values; only entries where
                ``mask == 1`` are used.
            mask: Binary observation mask of shape (N, D). 1 = observed,
                0 = missing.

        Raises:
            ValueError: If features and mask have incompatible shapes.
        """
        if features.shape != mask.shape:
            raise ValueError(
                f"features shape {features.shape} != mask shape {mask.shape}"
            )

        D = features.shape[1]
        self.scaler_means_ = np.zeros(D, dtype=np.float64)
        self.scaler_stds_ = np.ones(D, dtype=np.float64)

        for j in range(D):
            observed_idx = mask[:, j] > 0
            observed_vals = features[observed_idx, j]

            # Filter out any NaN that might sneak through
            observed_vals = observed_vals[~np.isnan(observed_vals)]

            if len(observed_vals) >= 2:
                self.scaler_means_[j] = np.mean(observed_vals)
                std = np.std(observed_vals)
                self.scaler_stds_[j] = std if std > 1e-8 else 1.0
            elif len(observed_vals) == 1:
                self.scaler_means_[j] = observed_vals[0]
                self.scaler_stds_[j] = 1.0
            else:
                # No observed values: leave mean=0, std=1
                logger.debug(
                    "Feature %d has no observed values; using default mean=0, std=1.",
                    j,
                )

    def standardize(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Z-score standardize features using the fitted scaler.

        Observed features are transformed as ``(x - mean) / std``.
        Missing features (where ``mask == 0``) are set to 0.0, which is
        the population mean after standardization. This ensures they
        contribute nothing to similarity computations.

        Args:
            features: Feature matrix of shape (N, D).
            mask: Binary observation mask of shape (N, D).

        Returns:
            Standardized feature matrix of shape (N, D) with missing
            values set to 0.0.

        Raises:
            RuntimeError: If :meth:`fit_scaler` has not been called.
        """
        if self.scaler_means_ is None or self.scaler_stds_ is None:
            raise RuntimeError("Scaler has not been fitted. Call fit_scaler() first.")

        # Vectorized standardization: (features - mean) / std * mask
        standardized = (
            (features - self.scaler_means_[np.newaxis, :])
            / self.scaler_stds_[np.newaxis, :]
        ) * mask

        # Replace any NaN resulting from 0/0 or inf
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

        return standardized.astype(np.float32)

    # ------------------------------------------------------------------
    # Pairwise similarity (exact loop)
    # ------------------------------------------------------------------

    def compute_pairwise_similarity(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        modality_labels: np.ndarray | None = None,
        modality_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute pairwise similarity using only mutually observed features.

        For each pair of patients (i, j):
            1. ``shared = mask[i] & mask[j]``  (element-wise AND)
            2. If ``sum(shared) < min_overlap``, similarity is 0.
            3. Otherwise, compute cosine similarity on the standardized
               shared features.
            4. Multiply by ``sqrt(num_shared / D)`` as an overlap penalty.
            5. Clamp negative similarities to 0 (only positive associations
               create edges).

        This is the exact O(N^2 * D) implementation. For large datasets,
        use :meth:`compute_pairwise_similarity_fast` instead.

        Args:
            features: Feature matrix of shape (N, D). Missing values may
                be NaN or arbitrary.
            mask: Binary observation mask of shape (N, D).
            modality_labels: Optional array of shape (D,) mapping each
                feature to a modality index. Used with ``modality_weights``
                to weight features by modality importance.
            modality_weights: Optional array of shape (M,) where M is the
                number of modalities. Weight applied to features from
                each modality during similarity computation.

        Returns:
            Symmetric similarity matrix of shape (N, N) with non-negative
            values. Diagonal is 0.
        """
        self._validate_inputs(features, mask, modality_labels, modality_weights)

        N, D = features.shape
        sim_matrix = np.zeros((N, N), dtype=np.float32)

        # Fit scaler if not already fitted
        if self.scaler_means_ is None:
            self.fit_scaler(features, mask)

        X = self.standardize(features, mask)

        # Convert mask to boolean for efficient indexing
        mask_bool = mask.astype(bool)

        for i in range(N):
            if not mask_bool[i].any():
                # Patient i has no observations at all
                continue

            for j in range(i + 1, N):
                shared = mask_bool[i] & mask_bool[j]
                num_shared = shared.sum()

                if num_shared < self.min_overlap:
                    continue

                xi = X[i, shared]
                xj = X[j, shared]

                # Apply per-modality weights if provided
                if modality_weights is not None and modality_labels is not None:
                    w = np.sqrt(modality_weights[modality_labels[shared]])
                    xi = xi * w
                    xj = xj * w

                # Cosine similarity
                norm_i = np.linalg.norm(xi)
                norm_j = np.linalg.norm(xj)

                if norm_i < 1e-8 or norm_j < 1e-8:
                    continue

                sim = float(np.dot(xi, xj) / (norm_i * norm_j))

                # Overlap penalty: penalize low-overlap pairs
                overlap_penalty = np.sqrt(num_shared / D)

                # Clamp to non-negative
                sim_matrix[i, j] = max(0.0, sim) * overlap_penalty
                sim_matrix[j, i] = sim_matrix[i, j]

        return sim_matrix

    # ------------------------------------------------------------------
    # Pairwise similarity (fast vectorized)
    # ------------------------------------------------------------------

    def compute_pairwise_similarity_fast(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        modality_labels: np.ndarray | None = None,
        modality_weights: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Vectorized pairwise similarity using batched matrix operations.

        This is functionally equivalent to :meth:`compute_pairwise_similarity`
        but uses numpy broadcasting and batched dot products to avoid the
        Python-level double loop. For N > 500 patients, this is typically
        10-50x faster at the cost of higher peak memory usage.

        The algorithm processes patient pairs in row-batches:
            1. For a batch of rows ``[i_start:i_end]``, compute the overlap
               matrix with all patients: ``overlap = mask[batch] @ mask.T``
            2. Compute the raw dot product: ``dots = X[batch] @ X.T``
            3. Compute norms accounting for shared features only (requires
               a mask-weighted norm computation).
            4. Combine: ``sim = dots / (norms_i * norms_j) * overlap_penalty``
            5. Apply min_overlap threshold and non-negativity.

        Args:
            features: Feature matrix of shape (N, D).
            mask: Binary observation mask of shape (N, D).
            modality_labels: Optional modality labels per feature, shape (D,).
            modality_weights: Optional per-modality weight array, shape (M,).
            batch_size: Number of rows to process per batch. Larger batches
                use more memory but may be faster. Default: 256.

        Returns:
            Symmetric similarity matrix of shape (N, N) with non-negative
            values. Diagonal is 0.

        Note:
            Memory usage is approximately O(batch_size * N) for intermediate
            matrices. For very large N (> 50000), consider reducing batch_size.
        """
        self._validate_inputs(features, mask, modality_labels, modality_weights)

        N, D = features.shape

        # Fit scaler if not already fitted
        if self.scaler_means_ is None:
            self.fit_scaler(features, mask)

        X = self.standardize(features, mask)

        # Apply modality weights to feature vectors if provided
        if modality_weights is not None and modality_labels is not None:
            feature_weights = np.sqrt(modality_weights[modality_labels])
            X = X * feature_weights[np.newaxis, :]  # (N, D)
            # Also weight the mask for consistent norm computation
            mask_weighted = mask * feature_weights[np.newaxis, :]
        else:
            mask_weighted = mask.copy()

        sim_matrix = np.zeros((N, N), dtype=np.float32)
        mask_f32 = mask.astype(np.float32)

        for i_start in range(0, N, batch_size):
            i_end = min(i_start + batch_size, N)
            batch_X = X[i_start:i_end]  # (B, D)
            batch_mask = mask_f32[i_start:i_end]  # (B, D)

            # Number of mutually observed features: (B, N)
            overlap_counts = batch_mask @ mask_f32.T

            # Raw dot product of standardized (and optionally weighted) features
            # Since unobserved values are 0 in X, the dot product naturally
            # only sums over mutually observed features: (B, N)
            raw_dots = batch_X @ X.T

            # Compute norms considering only mutually observed features.
            # For each (i, j), we need ||x_i[shared]|| and ||x_j[shared]||.
            # Trick: ||x_i[shared]||^2 = sum(x_i^2 * mask_j) for the shared subset.
            # (B, D) * (N, D).T won't work directly for per-pair norms.
            # Instead: x_i^2 @ mask_j.T gives sum of x_i^2 over shared features.
            X_sq = X**2
            batch_X_sq = X_sq[i_start:i_end]  # (B, D)

            # norm_i_sq[b, j] = sum of batch_X[b]^2 where mask[j] is 1
            norm_i_sq = batch_X_sq @ mask_f32.T  # (B, N)
            # norm_j_sq[b, j] = sum of X[j]^2 where batch_mask[b] is 1
            norm_j_sq = batch_mask @ X_sq.T  # (B, N)

            norm_i = np.sqrt(np.maximum(norm_i_sq, 1e-16))
            norm_j = np.sqrt(np.maximum(norm_j_sq, 1e-16))

            # Cosine similarity
            cos_sim = raw_dots / (norm_i * norm_j)

            # Overlap penalty: sqrt(num_shared / D)
            overlap_penalty = np.sqrt(overlap_counts / D)

            # Apply min_overlap threshold
            valid = overlap_counts >= self.min_overlap

            # Combine: clamp to non-negative, apply penalty and threshold
            batch_sim = np.maximum(cos_sim, 0.0) * overlap_penalty * valid

            sim_matrix[i_start:i_end, :] = batch_sim

        # Symmetrize: take the max of (i,j) and (j,i) to handle batch boundaries
        sim_matrix = np.maximum(sim_matrix, sim_matrix.T)

        # Zero the diagonal
        np.fill_diagonal(sim_matrix, 0.0)

        return sim_matrix

    # ------------------------------------------------------------------
    # kNN graph construction
    # ------------------------------------------------------------------

    def build_knn_graph(
        self,
        similarity_matrix: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a symmetric kNN graph from a pairwise similarity matrix.

        For each node, the top-k most similar neighbors (by similarity
        value) are selected. The graph is then symmetrized: if either
        i selects j or j selects i, an undirected edge is added between
        them. Both directed edges (i->j and j->i) are included with the
        same weight.

        Args:
            similarity_matrix: Symmetric matrix of shape (N, N) with
                non-negative similarity values. Diagonal should be 0.

        Returns:
            Tuple of:
                - ``edge_index``: Long tensor of shape (2, E) with source
                  and destination node indices.
                - ``edge_weight``: Float tensor of shape (E,) with edge
                  weights (similarity values).

        Raises:
            ValueError: If the similarity matrix is not square.
        """
        N = similarity_matrix.shape[0]
        if similarity_matrix.shape != (N, N):
            raise ValueError(
                f"Expected square matrix, got shape {similarity_matrix.shape}"
            )

        if N == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
            )

        # Collect directed kNN edges
        src_list = []
        dst_list = []
        weight_list = []

        for i in range(N):
            sims = similarity_matrix[i].copy()
            sims[i] = -np.inf  # exclude self-loop

            # Count how many positive similarities exist
            num_positive = (sims > 0).sum()

            if num_positive == 0:
                # Isolated node: no positive similarities at all
                logger.debug("Node %d is isolated (no positive similarities).", i)
                continue

            # Select top-k neighbors (or fewer if not enough positive)
            k = min(self.k_neighbors, num_positive)
            # Use argpartition for O(N) selection instead of O(N log N) sort
            top_k_indices = np.argpartition(sims, -k)[-k:]

            for j in top_k_indices:
                if sims[j] > 0:
                    src_list.append(i)
                    dst_list.append(j)
                    weight_list.append(float(sims[j]))

        if len(src_list) == 0:
            logger.warning(
                "No edges found in kNN graph. All patients may be isolated. "
                "Consider lowering min_overlap or increasing k_neighbors."
            )
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
            )

        # Symmetrize: ensure both directions exist with consistent weight
        edge_dict: dict[tuple[int, int], float] = {}
        for s, d, w in zip(src_list, dst_list, weight_list):
            key = (min(s, d), max(s, d))
            # Keep the maximum weight if an edge appears in both directions
            if key not in edge_dict or w > edge_dict[key]:
                edge_dict[key] = w

        final_src = []
        final_dst = []
        final_weight = []

        for (u, v), w in edge_dict.items():
            # Add both directions for undirected graph
            final_src.extend([u, v])
            final_dst.extend([v, u])
            final_weight.extend([w, w])

        edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)
        edge_weight = torch.tensor(final_weight, dtype=torch.float32)

        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # Overlap fractions
    # ------------------------------------------------------------------

    def compute_overlap_fractions(
        self,
        mask: np.ndarray,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the feature overlap fraction for each edge.

        The overlap fraction for edge (i, j) is defined as the number of
        mutually observed features divided by the total number of features:

            overlap_frac(i, j) = sum(mask[i] * mask[j]) / D

        This is used downstream in the availability-gated message passing
        layer to modulate message strength based on shared information.

        Args:
            mask: Binary observation mask of shape (N, D).
            edge_index: Edge index tensor of shape (2, E).

        Returns:
            Tensor of shape (E,) with overlap fractions in [0, 1].
        """
        if edge_index.shape[1] == 0:
            return torch.zeros(0, dtype=torch.float32)

        D = mask.shape[1]
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()

        # Vectorized overlap computation
        src_mask = mask[src]  # (E, D)
        dst_mask = mask[dst]  # (E, D)
        overlaps = (src_mask * dst_mask).sum(axis=1) / D  # (E,)

        return torch.tensor(overlaps, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build_full_graph(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        modality_labels: np.ndarray | None = None,
        modality_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Complete graph building pipeline.

        Steps:
            1. Fit the per-feature scaler on observed values.
            2. Compute pairwise similarity (auto-selects fast path if
               applicable).
            3. Build a symmetric kNN graph.
            4. Compute edge overlap fractions.

        Args:
            features: Feature matrix of shape (N, D).
            mask: Binary observation mask of shape (N, D).
            modality_labels: Optional per-feature modality indices, shape (D,).
            modality_weights: Optional per-modality weights, shape (M,).

        Returns:
            Dictionary with keys:
                - ``'edge_index'``: (2, E) long tensor.
                - ``'edge_weight'``: (E,) float tensor.
                - ``'overlap_frac'``: (E,) float tensor.
                - ``'similarity_matrix'``: (N, N) numpy array.
                - ``'num_nodes'``: int, number of patients.
                - ``'num_edges'``: int, number of directed edges.
                - ``'num_isolated'``: int, number of nodes with degree 0.
        """
        N = features.shape[0]
        logger.info(
            "Building partial-observation graph for %d patients, %d features.",
            N,
            features.shape[1],
        )

        # Step 1: Fit scaler
        self.fit_scaler(features, mask)

        # Step 2: Compute pairwise similarity
        if self.use_fast_path and N > self.fast_path_threshold:
            logger.info(
                "Using fast vectorized path (N=%d > threshold=%d).",
                N,
                self.fast_path_threshold,
            )
            sim_matrix = self.compute_pairwise_similarity_fast(
                features, mask, modality_labels, modality_weights
            )
        else:
            sim_matrix = self.compute_pairwise_similarity(
                features, mask, modality_labels, modality_weights
            )

        # Step 3: Build kNN graph
        edge_index, edge_weight = self.build_knn_graph(sim_matrix)

        # Step 4: Compute overlap fractions
        overlap_frac = self.compute_overlap_fractions(mask, edge_index)

        # Compute graph statistics
        num_edges = edge_index.shape[1]
        if num_edges > 0:
            connected_nodes = set(edge_index[0].numpy().tolist())
            num_isolated = N - len(connected_nodes)
        else:
            num_isolated = N

        if num_isolated > 0:
            logger.warning(
                "%d of %d patients are isolated (no graph neighbors). "
                "Consider lowering min_overlap (currently %d) or "
                "increasing k_neighbors (currently %d).",
                num_isolated,
                N,
                self.min_overlap,
                self.k_neighbors,
            )

        logger.info(
            "Graph built: %d nodes, %d directed edges, %d isolated nodes, "
            "mean degree %.1f.",
            N,
            num_edges,
            num_isolated,
            num_edges / max(N, 1),
        )

        return {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "overlap_frac": overlap_frac,
            "similarity_matrix": sim_matrix,
            "num_nodes": N,
            "num_edges": num_edges,
            "num_isolated": num_isolated,
        }

    # ------------------------------------------------------------------
    # Graph refinement with imputed features
    # ------------------------------------------------------------------

    def rebuild_with_imputed(
        self,
        observed_features: np.ndarray,
        imputed_features: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        modality_labels: np.ndarray | None = None,
        modality_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Rebuild the graph using a blend of observed and imputed features.

        During iterative graph refinement, the graph is rebuilt using a
        weighted combination of the original observed values and the
        current round's imputed values. For missing features, only the
        imputed values are used. For observed features, a convex
        combination controlled by ``alpha`` is used:

            blended[i, j] = alpha * observed[i, j] + (1 - alpha) * imputed[i, j]
                            if mask[i, j] == 1
            blended[i, j] = imputed[i, j]
                            if mask[i, j] == 0

        After blending, the mask becomes all-ones (every feature is now
        "available"), and the full graph pipeline is re-run.

        Args:
            observed_features: Original observed feature matrix, shape (N, D).
            imputed_features: Current imputed feature matrix, shape (N, D).
                Must have no NaN values.
            mask: Original binary observation mask, shape (N, D).
            alpha: Blending weight for observed features. Higher alpha
                means more trust in the original observations. Must be
                in [0, 1]. Default: 0.5.
            modality_labels: Optional per-feature modality indices.
            modality_weights: Optional per-modality weights.

        Returns:
            Same dictionary as :meth:`build_full_graph`.

        Raises:
            ValueError: If alpha is not in [0, 1] or shapes are inconsistent.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if observed_features.shape != imputed_features.shape:
            raise ValueError(
                f"observed shape {observed_features.shape} != "
                f"imputed shape {imputed_features.shape}"
            )
        if observed_features.shape != mask.shape:
            raise ValueError(
                f"features shape {observed_features.shape} != mask shape {mask.shape}"
            )

        if np.any(np.isnan(imputed_features)):
            warnings.warn(
                "imputed_features contains NaN values. These will be "
                "treated as 0 during standardization.",
                stacklevel=2,
            )

        # Blend observed and imputed
        blended = np.where(
            mask > 0,
            alpha * observed_features + (1.0 - alpha) * imputed_features,
            imputed_features,
        )

        # After blending, all features are now "available"
        full_mask = np.ones_like(mask)

        # Refit scaler on the blended data
        self.fit_scaler(blended, full_mask)

        logger.info("Rebuilding graph with alpha=%.2f (observed/imputed blend).", alpha)

        return self.build_full_graph(
            blended, full_mask, modality_labels, modality_weights
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def compute_similarity_for_new_patient(
        self,
        new_features: np.ndarray,
        new_mask: np.ndarray,
        existing_features: np.ndarray,
        existing_mask: np.ndarray,
        modality_labels: np.ndarray | None = None,
        modality_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute similarity between a single new patient and all existing patients.

        This is used for incremental graph updates when a new patient
        arrives, avoiding the O(N^2) full recomputation.

        Args:
            new_features: Feature vector for the new patient, shape (D,)
                or (1, D).
            new_mask: Observation mask for the new patient, shape (D,)
                or (1, D).
            existing_features: Feature matrix for existing patients,
                shape (N, D).
            existing_mask: Observation mask for existing patients,
                shape (N, D).
            modality_labels: Optional per-feature modality indices, shape (D,).
            modality_weights: Optional per-modality weights, shape (M,).

        Returns:
            Similarity vector of shape (N,) with non-negative values.
        """
        # Ensure 1D
        new_features = new_features.ravel()
        new_mask = new_mask.ravel()

        if self.scaler_means_ is None:
            raise RuntimeError(
                "Scaler not fitted. Call fit_scaler() or build_full_graph() first."
            )

        D = new_features.shape[0]
        N = existing_features.shape[0]

        # Standardize the new patient
        new_std = ((new_features - self.scaler_means_) / self.scaler_stds_) * new_mask
        new_std = np.nan_to_num(new_std, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

        # Standardize existing patients
        existing_std = self.standardize(existing_features, existing_mask)

        similarities = np.zeros(N, dtype=np.float32)

        # Vectorized overlap computation
        overlap_counts = (new_mask[np.newaxis, :] * existing_mask).sum(axis=1)  # (N,)

        for j in range(N):
            if overlap_counts[j] < self.min_overlap:
                continue

            shared = (new_mask > 0) & (existing_mask[j] > 0)
            xi = new_std[shared]
            xj = existing_std[j, shared]

            if modality_weights is not None and modality_labels is not None:
                w = np.sqrt(modality_weights[modality_labels[shared]])
                xi = xi * w
                xj = xj * w

            norm_i = np.linalg.norm(xi)
            norm_j = np.linalg.norm(xj)

            if norm_i < 1e-8 or norm_j < 1e-8:
                continue

            sim = float(np.dot(xi, xj) / (norm_i * norm_j))
            overlap_penalty = np.sqrt(overlap_counts[j] / D)
            similarities[j] = max(0.0, sim) * overlap_penalty

        return similarities

    def build(
        self,
        features: Any,
        mask: Any = None,
        k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a kNN graph (adapter for :class:`IterativeGraphRefiner`).

        Accepts torch tensors or numpy arrays and returns
        ``(edge_index, edge_weight)``.

        Args:
            features: Feature matrix ``(N, D)``.
            mask: Binary observation mask ``(N, D)``. If ``None``, all
                features are treated as observed.
            k: Override for ``k_neighbors`` (unused; kept for API compat).

        Returns:
            Tuple of ``(edge_index, edge_weight)``.
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if mask is None:
            mask = np.ones_like(features)

        features = features.astype(np.float32)
        mask = mask.astype(np.float32)

        result = self.build_full_graph(features, mask)
        return result["edge_index"], result["edge_weight"]

    def _validate_inputs(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        modality_labels: np.ndarray | None,
        modality_weights: np.ndarray | None,
    ) -> None:
        """Validate input array shapes and types.

        Raises:
            ValueError: If inputs are inconsistent.
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2-dimensional, got {features.ndim}D")
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2-dimensional, got {mask.ndim}D")
        if features.shape != mask.shape:
            raise ValueError(
                f"features shape {features.shape} != mask shape {mask.shape}"
            )
        if features.shape[0] == 0:
            raise ValueError("features must have at least 1 patient (row)")
        if features.shape[1] == 0:
            raise ValueError("features must have at least 1 feature (column)")

        if modality_labels is not None:
            if modality_labels.shape != (features.shape[1],):
                raise ValueError(
                    f"modality_labels shape {modality_labels.shape} "
                    f"!= (D={features.shape[1]},)"
                )
        if modality_weights is not None and modality_labels is None:
            raise ValueError(
                "modality_weights requires modality_labels to be provided."
            )
