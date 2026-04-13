"""Singleton model registry: loads all ML models once at FastAPI startup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from app.config import (
    CATBOOST_CHECKPOINT,
    CONFORMAL_AGGREGATE_PATH,
    DEFAULT_CONFORMAL_BAND_WIDTH,
    DEEPHIT_CHECKPOINT,
    GRAPHDT_CHECKPOINT,
)

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Select best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelRegistry:
    """Loads and holds all models for the dashboard."""

    def __init__(self) -> None:
        self.device: torch.device = _get_device()

        # CatBoost staging
        self.catboost_model = None

        # DeepHit survival
        self.deephit_model = None
        self.deephit_ckpt: dict = {}

        # Graph-DT survival
        self.graphdt_model = None
        self.graphdt_ckpt: dict = {}
        self.graphdt_node_enc: torch.Tensor | None = None

        # Conformal band width
        self.conformal_band_width: float = DEFAULT_CONFORMAL_BAND_WIDTH

        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_all(self) -> None:
        """Load all model checkpoints. Called once at startup."""
        logger.info(f"Loading models on device: {self.device}")

        self._load_catboost()
        self._load_deephit()
        self._load_graphdt()
        self._load_conformal_metadata()

        self._loaded = True
        logger.info("All models loaded successfully")

    def _load_catboost(self) -> None:
        """Load pre-trained CatBoost staging model."""
        if not CATBOOST_CHECKPOINT.exists():
            logger.warning(f"CatBoost checkpoint not found: {CATBOOST_CHECKPOINT}")
            logger.info("CatBoost will be trained on first request")
            return

        from catboost import CatBoostClassifier

        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model(str(CATBOOST_CHECKPOINT))
        logger.info(f"  CatBoost loaded from {CATBOOST_CHECKPOINT.name}")

    def _load_deephit(self) -> None:
        """Load DeepHit fold 0 checkpoint."""
        if not DEEPHIT_CHECKPOINT.exists():
            logger.warning(f"DeepHit checkpoint not found: {DEEPHIT_CHECKPOINT}")
            return

        from giman_pipeline.paper3.dynamic_deephit import load_deephit_checkpoint

        self.deephit_model, self.deephit_ckpt = load_deephit_checkpoint(
            DEEPHIT_CHECKPOINT, device=self.device
        )
        self.deephit_model.eval()
        logger.info(
            f"  DeepHit loaded: C-td={self.deephit_ckpt.get('fold_ctd', 'N/A')}, "
            f"input_dim={self.deephit_ckpt.get('input_dim', 'N/A')}"
        )

    def _load_graphdt(self) -> None:
        """Load Graph-DT fold 0 checkpoint and pre-compute GAT node embeddings."""
        if not GRAPHDT_CHECKPOINT.exists():
            logger.warning(f"Graph-DT checkpoint not found: {GRAPHDT_CHECKPOINT}")
            return

        from giman_pipeline.paper3.graph_digital_twin import load_graph_dt_checkpoint

        self.graphdt_model, self.graphdt_ckpt = load_graph_dt_checkpoint(
            GRAPHDT_CHECKPOINT, device=self.device
        )
        self.graphdt_model.eval()
        logger.info(
            f"  Graph-DT loaded: C-td={self.graphdt_ckpt.get('fold_ctd', 'N/A')}, "
            f"n_baseline={self.graphdt_ckpt.get('n_baseline_features', 'N/A')}"
        )

        # Pre-compute GAT node embeddings (one-time)
        self._precompute_gat_embeddings()

    def _precompute_gat_embeddings(self) -> None:
        """Run GAT forward pass on all graph nodes once at startup."""
        if self.graphdt_model is None:
            return

        node_baseline = self.graphdt_ckpt["node_baseline"].to(self.device)
        edge_index = self.graphdt_ckpt["edge_index"].to(self.device)

        with torch.no_grad():
            node_enc = self.graphdt_model.node_encoder(node_baseline)
            for gat_layer in self.graphdt_model.gat_layers_list:
                node_enc = gat_layer(node_enc, edge_index)
            node_enc = self.graphdt_model.gat_norm(self.graphdt_model.gat_proj(node_enc))

        self.graphdt_node_enc = node_enc
        logger.info(f"  GAT node embeddings pre-computed: {node_enc.shape}")

    def _load_conformal_metadata(self) -> None:
        """Load conformal band width from Paper 4 aggregate summary."""
        if not CONFORMAL_AGGREGATE_PATH.exists():
            logger.warning("Conformal aggregate not found, using default band width")
            return

        with open(CONFORMAL_AGGREGATE_PATH) as f:
            agg = json.load(f)

        # Find DeepHit 90% CL band width
        for entry in agg.get("per_model", []):
            if (
                entry.get("model") == "deephit"
                and abs(entry.get("confidence_level", 0) - 0.90) < 0.01
            ):
                self.conformal_band_width = entry.get("mean_band_width", DEFAULT_CONFORMAL_BAND_WIDTH)
                break

        logger.info(f"  Conformal band width: {self.conformal_band_width:.4f}")
