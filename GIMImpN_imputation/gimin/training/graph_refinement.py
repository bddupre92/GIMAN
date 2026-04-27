"""Iterative graph refinement for GIMIN training.

During training the patient similarity graph is rebuilt periodically using
a blend of observed and imputed features.  As training progresses and the
imputation quality improves, the blending coefficient *alpha* decreases so
that the graph increasingly reflects model-imputed values.

Typical schedule (3 refinement rounds over 200 epochs):

    Round 1  (epochs   1 -- 70): alpha = 1.00  (graph built from observed only)
    Round 2  (epochs  71 -- 140): alpha = 0.75
    Round 3  (epochs 141 -- 200): alpha = 0.50
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class IterativeGraphRefiner:
    """Manages iterative graph rebuilding during GIMIN training.

    The refiner linearly interpolates a blending coefficient *alpha* from
    ``alpha_start`` to ``alpha_end`` across refinement rounds.  At each
    round the patient feature matrix is blended:

    .. math::

        x_{\\text{blend}} = \\alpha \\cdot x_{\\text{obs}} +
                            (1 - \\alpha) \\cdot x_{\\text{imp}}

    For positions where values are genuinely missing (``mask == 0``), the
    imputed value is always used regardless of *alpha*.  The blended
    features are then passed to the graph builder to construct a new kNN
    graph.

    Args:
        graph_builder: A callable (or object with a ``build`` method) that
            accepts a feature matrix and returns ``(edge_index, edge_weight)``.
            Expected signature:
            ``build(features, mask=None, k=None) -> (edge_index, edge_weight)``
        alpha_start: Initial blending coefficient (1.0 = observed only).
            Default: 1.0.
        alpha_end: Final blending coefficient. Default: 0.5.
        num_rounds: Total number of refinement rounds. Default: 3.
    """

    def __init__(
        self,
        graph_builder: object,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        num_rounds: int = 3,
    ) -> None:
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
        if not (0.0 <= alpha_end <= alpha_start <= 1.0):
            raise ValueError(
                f"Require 0 <= alpha_end <= alpha_start <= 1, "
                f"got alpha_start={alpha_start}, alpha_end={alpha_end}"
            )

        self.graph_builder = graph_builder
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.num_rounds = num_rounds

        self._current_round: int = 0

    # ------------------------------------------------------------------
    # Alpha schedule
    # ------------------------------------------------------------------

    def get_alpha(self, current_round: int) -> float:
        """Compute the blending coefficient for a given refinement round.

        Linearly interpolates from ``alpha_start`` (round 0) to
        ``alpha_end`` (round ``num_rounds - 1``).

        Args:
            current_round: Zero-indexed refinement round number.

        Returns:
            Blending coefficient in ``[alpha_end, alpha_start]``.
        """
        if self.num_rounds <= 1:
            return self.alpha_start
        progress = min(current_round, self.num_rounds - 1) / (self.num_rounds - 1)
        alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
        return alpha

    # ------------------------------------------------------------------
    # Feature blending
    # ------------------------------------------------------------------

    @staticmethod
    def blend_features(
        observed_features: torch.Tensor,
        imputed_features: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Blend observed and imputed features for graph construction.

        Where data is observed (``mask == 1``):

        .. math::

            x_{\\text{blend}} = \\alpha \\cdot x_{\\text{obs}} +
                                (1 - \\alpha) \\cdot x_{\\text{imp}}

        Where data is missing (``mask == 0``):

        .. math::

            x_{\\text{blend}} = x_{\\text{imp}}

        Args:
            observed_features: Original feature matrix with arbitrary
                values at missing positions, shape ``(N, F)``.
            imputed_features: Model-imputed feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.
            alpha: Blending weight for observed values.

        Returns:
            Blended feature matrix, shape ``(N, F)``.
        """
        blended = imputed_features.clone()
        observed_positions = mask.bool()

        # Where observed: weighted average of observed and imputed.
        blended[observed_positions] = (
            alpha * observed_features[observed_positions]
            + (1.0 - alpha) * imputed_features[observed_positions]
        )
        return blended

    # ------------------------------------------------------------------
    # Graph rebuild
    # ------------------------------------------------------------------

    def rebuild_graph(
        self,
        observed_features: torch.Tensor,
        imputed_features: torch.Tensor,
        mask: torch.Tensor,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend features and rebuild the kNN patient similarity graph.

        Args:
            observed_features: Original feature matrix, shape ``(N, F)``.
            imputed_features: Model-imputed feature matrix, shape ``(N, F)``.
            mask: Binary observation mask, shape ``(N, F)``.
            alpha: Override blending coefficient.  If ``None``, computed
                from the current round via :meth:`get_alpha`.

        Returns:
            Tuple of ``(edge_index, edge_weight)`` as produced by the
            underlying graph builder.
        """
        if alpha is None:
            alpha = self.get_alpha(self._current_round)

        logger.info(
            "Rebuilding graph at round %d with alpha=%.3f",
            self._current_round,
            alpha,
        )

        blended = self.blend_features(observed_features, imputed_features, mask, alpha)

        # Call graph builder.  Supports both callable and objects with
        # a ``build`` method.
        if hasattr(self.graph_builder, "build"):
            edge_index, edge_weight = self.graph_builder.build(blended, mask=mask)
        elif callable(self.graph_builder):
            edge_index, edge_weight = self.graph_builder(blended, mask=mask)
        else:
            raise TypeError(
                f"graph_builder must be callable or have a 'build' method, "
                f"got {type(self.graph_builder)}"
            )

        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------

    def advance_round(self) -> int:
        """Advance to the next refinement round.

        Returns:
            The new current round number.
        """
        self._current_round += 1
        logger.info(
            "Advanced to refinement round %d / %d (alpha=%.3f)",
            self._current_round,
            self.num_rounds,
            self.get_alpha(self._current_round),
        )
        return self._current_round

    def reset(self) -> None:
        """Reset the refiner to round 0."""
        self._current_round = 0

    @property
    def current_round(self) -> int:
        """Current refinement round (zero-indexed)."""
        return self._current_round
