"""Inductive Graph Extension for Temporal Validation.

When Graph-DT is trained on a temporal window, the kNN patient similarity
graph only contains training patients. Test patients (enrolled later) are
absent from the graph. This module extends the training graph inductively
by connecting each test patient to its k nearest training neighbors.

Key properties:
    - GAT is inherently inductive (Velickovic et al., 2018): shared edge-wise
      attention weights generalize to unseen nodes.
    - Test patients receive aggregated messages FROM training neighbors.
    - NO edges between test patients (prevents test-test leakage).
    - Self-loops on test nodes for GAT stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from giman_pipeline.paper3.graph_digital_twin import GRAPH_FEATURES


class InductiveGraphExtender:
    """Extends a training-only kNN graph to include test patients.

    Algorithm:
        1. Append test node features after training nodes
        2. For each test patient: compute cosine similarity to all training nodes
        3. Connect test patient → k nearest training neighbors (directed)
        4. Add self-loops on test nodes for GAT message-passing stability
        5. Do NOT add test-test edges (prevents future-patient leakage)
    """

    def __init__(
        self,
        train_node_baseline: torch.Tensor,
        train_edge_index: torch.Tensor,
        train_edge_weight: torch.Tensor,
        train_pat_to_gidx: dict[int, int],
        k_neighbors: int = 15,
    ):
        """Args:
        train_node_baseline: (N_train, n_features) standardized baseline features
        train_edge_index: (2, E_train) edge indices for training graph
        train_edge_weight: (E_train,) cosine similarity weights
        train_pat_to_gidx: {patno: node_index} for training patients
        k_neighbors: Number of training neighbors per test node
        """
        self.train_node_baseline = train_node_baseline
        self.train_edge_index = train_edge_index
        self.train_edge_weight = train_edge_weight
        self.train_pat_to_gidx = train_pat_to_gidx
        self.k_neighbors = k_neighbors
        self.n_train = train_node_baseline.size(0)

    def extend_for_test(
        self,
        test_baseline_features: torch.Tensor,
        test_patnos: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
        """Extend the training graph with test patient nodes.

        Args:
            test_baseline_features: (N_test, n_features) standardized features
            test_patnos: Patient IDs for test nodes (order matches features)

        Returns:
            extended_node_baseline: (N_train + N_test, n_features)
            extended_edge_index: (2, E_extended) with train-train + test→train edges
            extended_edge_weight: (E_extended,) cosine similarity weights
            full_pat_to_gidx: {patno: node_index} for ALL patients
        """
        n_test = test_baseline_features.size(0)
        n_total = self.n_train + n_test

        # 1. Concatenate node features: training first, then test
        extended_baseline = torch.cat(
            [self.train_node_baseline, test_baseline_features],
            dim=0,
        )

        # 2. Build pat_to_gidx for all patients
        full_pat_to_gidx = dict(self.train_pat_to_gidx)
        for i, patno in enumerate(test_patnos):
            full_pat_to_gidx[patno] = self.n_train + i

        # 3. Compute test→training edges via cosine similarity
        new_src, new_dst, new_weights = [], [], []

        # Normalize for cosine similarity
        train_normed = self.train_node_baseline.numpy()
        norms_tr = np.linalg.norm(train_normed, axis=1, keepdims=True)
        norms_tr[norms_tr < 1e-8] = 1.0
        train_normed = train_normed / norms_tr

        test_normed = test_baseline_features.numpy()
        norms_te = np.linalg.norm(test_normed, axis=1, keepdims=True)
        norms_te[norms_te < 1e-8] = 1.0
        test_normed = test_normed / norms_te

        # Cosine similarity: (N_test, N_train)
        sim_matrix = test_normed @ train_normed.T

        k = min(self.k_neighbors, self.n_train - 1)

        for i in range(n_test):
            test_gidx = self.n_train + i
            sims = sim_matrix[i]

            # Top-k training neighbors
            topk_indices = np.argpartition(sims, -k)[-k:]
            for train_gidx in topk_indices:
                w = float(sims[train_gidx])
                if w > 0:
                    # Bidirectional edge: test ↔ training neighbor
                    new_src.append(test_gidx)
                    new_dst.append(int(train_gidx))
                    new_weights.append(w)
                    new_src.append(int(train_gidx))
                    new_dst.append(test_gidx)
                    new_weights.append(w)

            # Self-loop for GAT stability
            new_src.append(test_gidx)
            new_dst.append(test_gidx)
            new_weights.append(1.0)

        # 4. Combine training edges + new test→train edges
        if new_src:
            new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
            new_edge_weight = torch.tensor(new_weights, dtype=torch.float32)

            extended_edge_index = torch.cat(
                [self.train_edge_index, new_edge_index],
                dim=1,
            )
            extended_edge_weight = torch.cat(
                [self.train_edge_weight, new_edge_weight],
                dim=0,
            )
        else:
            extended_edge_index = self.train_edge_index
            extended_edge_weight = self.train_edge_weight

        return (
            extended_baseline,
            extended_edge_index,
            extended_edge_weight,
            full_pat_to_gidx,
        )

    def get_stats(self, n_test: int) -> dict:
        """Summary statistics about the extended graph."""
        return {
            "n_train_nodes": self.n_train,
            "n_test_nodes": n_test,
            "n_total_nodes": self.n_train + n_test,
            "n_train_edges": int(self.train_edge_index.size(1)),
            "k_neighbors": self.k_neighbors,
            "max_new_edges_per_test": 2 * self.k_neighbors
            + 1,  # bidirectional + self-loop
        }


def build_inductive_extender(
    features_df: pd.DataFrame,
    train_patnos: list[int],
    k_neighbors: int = 15,
) -> InductiveGraphExtender:
    """Build an InductiveGraphExtender from training patients' baseline features.

    Convenience function that wraps build_patient_graph() from Paper 3 and
    constructs an extender ready to accept test patients.

    Args:
        features_df: Longitudinal features DataFrame
        train_patnos: Patient IDs in the training set
        k_neighbors: kNN graph parameter

    Returns:
        InductiveGraphExtender with training graph pre-built
    """
    from giman_pipeline.paper3.graph_digital_twin import build_patient_graph

    edge_index, edge_weight, node_baseline = build_patient_graph(
        features_df,
        train_patnos,
        k_neighbors=k_neighbors,
    )
    pat_to_gidx = {p: i for i, p in enumerate(train_patnos)}

    return InductiveGraphExtender(
        train_node_baseline=node_baseline,
        train_edge_index=edge_index,
        train_edge_weight=edge_weight,
        train_pat_to_gidx=pat_to_gidx,
        k_neighbors=k_neighbors,
    )


def extract_test_baseline_features(
    features_df: pd.DataFrame,
    test_patnos: list[int],
    train_means: np.ndarray | None = None,
    train_stds: np.ndarray | None = None,
) -> torch.Tensor:
    """Extract and standardize baseline features for test patients.

    Uses the same GRAPH_FEATURES as build_patient_graph() from Paper 3.
    Standardization uses training statistics to prevent information leakage.

    Args:
        features_df: Longitudinal features DataFrame
        test_patnos: Test patient IDs (order preserved in output)
        train_means: Per-feature means from training set. If None, uses test set.
        train_stds: Per-feature stds from training set.

    Returns:
        (N_test, n_graph_features) float tensor, standardized
    """
    baseline = features_df[features_df["months_from_baseline"] == 0.0]
    baseline = baseline[baseline["PATNO"].isin(test_patnos)]

    pat_to_idx = {p: i for i, p in enumerate(test_patnos)}
    cols = [c for c in GRAPH_FEATURES if c in baseline.columns]
    n = len(test_patnos)

    feat_matrix = np.zeros((n, len(cols)), dtype=np.float32)
    mask = np.zeros((n, len(cols)), dtype=bool)

    for _, row in baseline.iterrows():
        idx = pat_to_idx.get(int(row["PATNO"]))
        if idx is None:
            continue
        for ci, col in enumerate(cols):
            val = row[col]
            if pd.notna(val):
                feat_matrix[idx, ci] = float(val)
                mask[idx, ci] = True

    # Standardize using training statistics
    if train_means is not None and train_stds is not None:
        for ci in range(len(cols)):
            if train_stds[ci] > 1e-8:
                feat_matrix[:, ci] = (
                    feat_matrix[:, ci] - train_means[ci]
                ) / train_stds[ci]
            else:
                feat_matrix[:, ci] = 0.0
    else:
        # Fallback: standardize from observed test values
        for ci in range(len(cols)):
            obs = feat_matrix[mask[:, ci], ci]
            if len(obs) > 1:
                m, s = obs.mean(), obs.std()
                if s > 1e-8:
                    feat_matrix[:, ci] = (feat_matrix[:, ci] - m) / s
                else:
                    feat_matrix[:, ci] = 0.0

    feat_matrix[~mask] = 0.0
    return torch.tensor(feat_matrix, dtype=torch.float32)
