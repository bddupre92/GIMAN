"""Main training loop for the GIMIN framework.

Orchestrates:

1. Masked-value self-supervision -- a fraction of observed values are
   artificially hidden each epoch and the model must reconstruct them.
2. Iterative graph refinement -- the patient similarity graph is rebuilt
   at configurable intervals using blended observed/imputed features.
3. Checkpoint management -- model and optimizer state can be saved and
   restored for long-running experiments.

Typical usage::

    from gimin.config import GIMINConfig
    from gimin.model.gimin_core import GIMIN
    from gimin.training.trainer import GIMINTrainer

    config = GIMINConfig.from_yaml("configs/default.yaml")
    model = GIMIN(config)
    trainer = GIMINTrainer(model, config, graph_builder)
    trainer.fit(dataset)
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config import GIMINConfig
from .graph_refinement import IterativeGraphRefiner
from .losses import GIMINLoss

logger = logging.getLogger(__name__)


class GIMINTrainer:
    """Main training loop for GIMIN.

    Uses AdamW optimisation with cosine annealing learning-rate scheduling.
    Each epoch randomly masks a fraction of observed values and trains the
    model to reconstruct them while maintaining distributional fidelity and
    cross-modal consistency.

    Args:
        model: A :class:`~gimin.model.gimin_core.GIMIN` instance.
        config: A :class:`~gimin.config.GIMINConfig` instance containing
            all training hyperparameters.
        graph_builder: An object (or callable) capable of constructing the
            patient similarity graph.  Passed to
            :class:`~gimin.training.graph_refinement.IterativeGraphRefiner`.
        cross_modal_pairs: Optional list of ``(feat_idx_a, feat_idx_b)``
            tuples for the cross-modal consistency loss.
        device: Torch device.  Default: auto-detect GPU.
    """

    def __init__(
        self,
        model: nn.Module,
        config: GIMINConfig,
        graph_builder: object,
        cross_modal_pairs: list[tuple[int, int]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)

        # Optimiser and scheduler.
        tp = config.training
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=tp.lr,
            weight_decay=tp.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tp.num_epochs, eta_min=1e-6
        )

        # Loss function.
        self.criterion = GIMINLoss(
            lambda_dist=tp.lambda_dist,
            lambda_cross=tp.lambda_cross,
            cross_modal_pairs=cross_modal_pairs,
        )

        # Graph refinement.
        gp = config.graph
        self.graph_refiner = IterativeGraphRefiner(
            graph_builder=graph_builder,
            alpha_start=gp.alpha_start,
            alpha_end=gp.alpha_end,
            num_rounds=gp.graph_refinement_iterations,
        )

        # Modality dimensions for forward pass.
        self.modality_dims = [len(mod.features) for mod in config.modalities]

        # Training state.
        self._best_loss: float = float("inf")
        self._best_model_state: dict[str, Any] | None = None
        self._epochs_without_improvement: int = 0
        self._history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def create_masked_batch(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        mask_fraction: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask a fraction of observed values for self-supervision.

        Args:
            features: Original feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.
            mask_fraction: Fraction of observed values to hide.
                Default: 0.2.

        Returns:
            Tuple of ``(masked_features, training_mask, target_mask)``:

            - ``masked_features``: Features with artificially masked values
              zeroed out.
            - ``training_mask``: Updated observation mask with artificial
              masks applied (used as model input).
            - ``target_mask``: Binary mask marking *only* the artificially
              hidden values (loss computed here).
        """
        # Identify observed positions.
        observed_positions = mask.bool()

        # Sample a random subset of observed positions to mask.
        rand_vals = torch.rand_like(features)
        target_mask = (rand_vals < mask_fraction) & observed_positions
        target_mask = target_mask.float()

        # Build training mask: original mask minus artificially masked.
        training_mask = mask.clone()
        training_mask[target_mask.bool()] = 0.0

        # Zero out artificially masked values in the feature matrix.
        masked_features = features.clone()
        masked_features[target_mask.bool()] = 0.0

        return masked_features, training_mask, target_mask

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        overlap_frac: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one training epoch.

        Args:
            features: Full (uncorrupted) feature matrix, shape ``(N, F)``.
            mask: Binary observation mask, shape ``(N, F)``.
            edge_index: Graph edge indices, shape ``(2, E)``.
            edge_weight: Edge weights, shape ``(E,)``.
            overlap_frac: Optional per-edge overlap fraction used by the
                availability-gated message passing layers.

        Returns:
            Dictionary of per-component loss values for this epoch.
        """
        self.model.train()

        # Move tensors to device.
        features = features.to(self.device)
        mask = mask.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        if overlap_frac is not None:
            overlap_frac = overlap_frac.to(self.device)

        # Create self-supervised masks.
        masked_features, training_mask, target_mask = self.create_masked_batch(
            features, mask, mask_fraction=self.config.training.batch_mask_fraction
        )

        # Forward pass.
        model_input = {
            "features": masked_features,
            "mask": training_mask,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "overlap_frac": overlap_frac
            if overlap_frac is not None
            else torch.ones(edge_index.shape[1], device=self.device),
            "modality_dims": self.modality_dims,
        }

        model_output = self.model(**model_input)

        # Compute loss.
        loss_dict = self.criterion(
            model_output=model_output,
            true_values=features,
            target_mask=target_mask,
            observed_mask=mask,
        )

        # Backward pass and optimise.
        self.optimizer.zero_grad()
        loss_dict["total"].backward()

        # Gradient clipping for stability.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Convert to plain floats for logging.
        return {k: v.item() for k, v in loss_dict.items()}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: dict[str, torch.Tensor],
        num_epochs: int | None = None,
        num_refinement_rounds: int | None = None,
    ) -> list[dict[str, float]]:
        """Full training with iterative graph refinement.

        The training is divided into ``num_refinement_rounds`` phases.
        At the start of each phase the patient similarity graph is rebuilt
        using blended features with a decreasing alpha coefficient.

        Default schedule (3 rounds, 200 epochs):

        - Round 1: epochs   1 --  70, alpha = 1.00
        - Round 2: epochs  71 -- 140, alpha = 0.75
        - Round 3: epochs 141 -- 200, alpha = 0.50

        Args:
            dataset: Dictionary containing at least:

                - ``"features"``: shape ``(N, F)``
                - ``"mask"``: shape ``(N, F)``
                - ``"edge_index"``: shape ``(2, E)``  (optional, will be
                  built if absent)
                - ``"edge_weight"``: shape ``(E,)``  (optional)
                - ``"overlap_frac"``: shape ``(E,)``  (optional)
            num_epochs: Override total epoch count.  If ``None``, uses
                ``config.training.num_epochs``.
            num_refinement_rounds: Override refinement round count.
                If ``None``, uses ``config.graph.graph_refinement_iterations``.

        Returns:
            Training history -- list of per-epoch loss dictionaries.
        """
        num_epochs = num_epochs or self.config.training.num_epochs
        num_refinement_rounds = (
            num_refinement_rounds or self.config.graph.graph_refinement_iterations
        )
        patience = self.config.training.early_stopping_patience

        features = dataset["features"]
        mask = dataset["mask"]

        # Initial graph (may be provided or built from scratch).
        if "edge_index" in dataset and dataset["edge_index"] is not None:
            edge_index = dataset["edge_index"]
            edge_weight = dataset.get(
                "edge_weight",
                torch.ones(dataset["edge_index"].shape[1]),
            )
        else:
            edge_index, edge_weight = self.graph_refiner.rebuild_graph(
                observed_features=features,
                imputed_features=torch.zeros_like(features),
                mask=mask,
                alpha=1.0,
            )

        overlap_frac = dataset.get("overlap_frac", None)

        # Compute epochs per refinement round.
        epochs_per_round = math.ceil(num_epochs / num_refinement_rounds)

        self.graph_refiner.reset()
        self._history = []
        self._epochs_without_improvement = 0
        self._best_loss = float("inf")
        self._best_model_state = None

        logger.info(
            "Starting GIMIN training: %d epochs, %d refinement rounds "
            "(%d epochs/round)",
            num_epochs,
            num_refinement_rounds,
            epochs_per_round,
        )

        global_epoch = 0

        for round_idx in range(num_refinement_rounds):
            alpha = self.graph_refiner.get_alpha(round_idx)
            logger.info(
                "=== Refinement round %d/%d  (alpha=%.3f) ===",
                round_idx + 1,
                num_refinement_rounds,
                alpha,
            )

            # Rebuild graph (skip for round 0 if we already have edges).
            if round_idx > 0:
                with torch.no_grad():
                    self.model.eval()
                    _overlap = (
                        overlap_frac
                        if overlap_frac is not None
                        else torch.ones(edge_index.shape[1])
                    )
                    model_output = self.model(
                        features=features.to(self.device),
                        mask=mask.to(self.device),
                        edge_index=edge_index.to(self.device),
                        edge_weight=edge_weight.to(self.device),
                        overlap_frac=_overlap.to(self.device),
                        modality_dims=self.modality_dims,
                    )
                    imputed = model_output["imputed"].cpu()

                edge_index, edge_weight = self.graph_refiner.rebuild_graph(
                    observed_features=features,
                    imputed_features=imputed,
                    mask=mask,
                    alpha=alpha,
                )
                self.graph_refiner.advance_round()

            # Train for this round's epochs.
            round_start_epoch = global_epoch
            round_end_epoch = min(round_start_epoch + epochs_per_round, num_epochs)
            early_stopped = False

            for epoch in range(round_start_epoch, round_end_epoch):
                t0 = time.time()
                epoch_losses = self.train_epoch(
                    features=features,
                    mask=mask,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    overlap_frac=overlap_frac,
                )
                self.scheduler.step()
                elapsed = time.time() - t0

                self._history.append(epoch_losses)
                global_epoch += 1

                # Logging.
                if (epoch + 1) % 10 == 0 or epoch == round_start_epoch:
                    logger.info(
                        "Epoch %3d/%d  |  loss=%.4f  recon=%.4f  "
                        "dist=%.4f  cross=%.4f  |  lr=%.2e  |  %.1fs",
                        epoch + 1,
                        num_epochs,
                        epoch_losses["total"],
                        epoch_losses["reconstruction"],
                        epoch_losses["distribution"],
                        epoch_losses["cross_modal"],
                        self.optimizer.param_groups[0]["lr"],
                        elapsed,
                    )

                # Early stopping check.
                current_loss = epoch_losses["total"]
                if current_loss < self._best_loss:
                    self._best_loss = current_loss
                    self._best_model_state = copy.deepcopy(self.model.state_dict())
                    self._epochs_without_improvement = 0
                else:
                    self._epochs_without_improvement += 1

                if self._epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d "
                        "(patience=%d, best_loss=%.4f)",
                        epoch + 1,
                        patience,
                        self._best_loss,
                    )
                    early_stopped = True
                    break

            if early_stopped:
                break

        # Restore best model weights.
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            logger.info("Restored best model weights (loss=%.4f).", self._best_loss)

        return self._history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model, optimiser, scheduler, and training state.

        Args:
            path: File path for the checkpoint (e.g.,
                ``"outputs/checkpoints/gimin_epoch200.pt"``).
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self._best_loss,
            "epochs_without_improvement": self._epochs_without_improvement,
            "history": self._history,
            "graph_refiner_round": self.graph_refiner.current_round,
        }
        torch.save(checkpoint, filepath)
        logger.info("Checkpoint saved to %s", filepath)

    def load_checkpoint(self, path: str) -> None:
        """Load model, optimiser, scheduler, and training state.

        Args:
            path: File path to a previously saved checkpoint.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._best_loss = checkpoint.get("best_loss", float("inf"))
        self._epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        self._history = checkpoint.get("history", [])

        round_num = checkpoint.get("graph_refiner_round", 0)
        self.graph_refiner.reset()
        for _ in range(round_num):
            self.graph_refiner.advance_round()

        logger.info("Checkpoint loaded from %s", filepath)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, float]]:
        """Training history: list of per-epoch loss dictionaries."""
        return self._history

    @property
    def best_loss(self) -> float:
        """Best (lowest) total loss observed during training."""
        return self._best_loss
