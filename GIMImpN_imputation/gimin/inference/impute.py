"""Production imputer using a trained GIMIN model.

Provides a high-level API for batch imputation with optional MC dropout
uncertainty estimation.  Supports loading from a checkpoint, imputing
raw NumPy arrays or PyTorch tensors, and converting results back to
pandas DataFrames.

Typical usage::

    from gimin.config import GIMINConfig
    from gimin.inference.impute import GIMINImputer

    config = GIMINConfig.from_yaml("configs/default.yaml")
    imputer = GIMINImputer("outputs/checkpoints/best.pt", config)

    imputed_df = imputer.impute_to_dataframe(
        df=patient_df,
        feature_columns=config.all_feature_names,
        return_uncertainty=True,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import GIMINConfig
from ..utils import ArrayLike

logger = logging.getLogger(__name__)


class GIMINImputer:
    """Production imputer using a trained GIMIN model.

    Loads a model checkpoint, manages device placement, and provides
    both low-level tensor-based and high-level DataFrame-based
    imputation interfaces.

    Args:
        model_path: Path to a saved model checkpoint (as produced by
            :meth:`~gimin.training.trainer.GIMINTrainer.save_checkpoint`).
        config: GIMIN configuration.  Must match the architecture used
            during training.
        model: Optional pre-constructed model instance.  If ``None``,
            the model is instantiated from *config* and weights are
            loaded from *model_path*.
        device: Torch device.  Default: auto-detect GPU.
        mc_samples: Number of MC dropout forward passes for uncertainty
            estimation.  Default: 50.
    """

    def __init__(
        self,
        model_path: str,
        config: GIMINConfig,
        model: nn.Module | None = None,
        device: torch.device | None = None,
        mc_samples: int = 50,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mc_samples = mc_samples
        self._model_path = Path(model_path)

        # Instantiate or use the provided model.
        if model is not None:
            self.model = model.to(self.device)
            self.scaler = None
        else:
            self.model = self._load_model(model_path, config)

        # Cached graph state (edge_index, edge_weight) for inference.
        self._edge_index: torch.Tensor | None = None
        self._edge_weight: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, config: GIMINConfig) -> nn.Module:
        """Instantiate the GIMIN model and load checkpoint weights.

        Args:
            model_path: Path to the checkpoint file.
            config: GIMIN configuration.

        Returns:
            Model with loaded weights, placed on *self.device*.

        Raises:
            FileNotFoundError: If *model_path* does not exist.
            ImportError: If the GIMIN model class cannot be imported.
        """
        filepath = Path(model_path)
        if not filepath.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

        # Import the GIMIN model class.
        from ..model.gimin_core import GIMIN

        model = GIMIN(
            modality_dims=config.modality_dims,
            embed_dim=config.model.embed_dim,
            num_gnn_layers=config.model.num_gnn_layers,
            num_heads=config.model.num_heads,
            mc_dropout=config.model.mc_dropout_rate,
            binary_feature_indices=getattr(config, "binary_feature_indices", None),
        )
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Support both full-checkpoint and state-dict-only formats.
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        # Load scaler from checkpoint if present.
        if isinstance(checkpoint, dict) and "scaler_state_dict" in checkpoint:
            from ..data.scaler import ModalityAwareScaler

            self.scaler = ModalityAwareScaler()
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("Loaded scaler from checkpoint.")
        else:
            self.scaler = None

        logger.info("Loaded GIMIN model from %s", filepath)
        return model

    # ------------------------------------------------------------------
    # Graph state management
    # ------------------------------------------------------------------

    def set_graph(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> None:
        """Set the patient similarity graph for inference.

        Args:
            edge_index: Edge index tensor, shape ``(2, E)``.
            edge_weight: Optional edge weights, shape ``(E,)``.
        """
        self._edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            self._edge_weight = edge_weight.to(self.device)
        else:
            self._edge_weight = torch.ones(edge_index.shape[1], device=self.device)

    def load_graph_state(self, path: str) -> None:
        """Load a saved graph state from disk.

        Args:
            path: Path to a saved graph state file (produced by
                ``torch.save({"edge_index": ..., "edge_weight": ...}, path)``).

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Graph state not found: {filepath}")

        state = torch.load(filepath, map_location=self.device, weights_only=True)
        self._edge_index = state["edge_index"].to(self.device)
        self._edge_weight = state.get("edge_weight")
        if self._edge_weight is not None:
            self._edge_weight = self._edge_weight.to(self.device)
        logger.info("Graph state loaded from %s", filepath)

    # ------------------------------------------------------------------
    # Core imputation
    # ------------------------------------------------------------------

    def impute(
        self,
        features: ArrayLike,
        mask: ArrayLike,
        return_uncertainty: bool = True,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> dict[str, np.ndarray]:
        """Batch imputation with optional MC dropout uncertainty.

        If ``return_uncertainty=True``, performs ``self.mc_samples``
        stochastic forward passes with dropout enabled and returns
        the mean prediction, per-value standard deviation, and the
        model's own predicted log-variance.

        Args:
            features: Feature matrix, shape ``(N, F)``.  Missing values
                should be set to 0 (they are masked out by *mask*).
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.
            return_uncertainty: If ``True``, run MC dropout and return
                uncertainty estimates.  Default: ``True``.
            edge_index: Optional override for the graph edge indices.
            edge_weight: Optional override for the graph edge weights.

        Returns:
            Dictionary with keys:

            - ``"imputed"``: imputed feature matrix, shape ``(N, F)``
            - ``"pred_mean"``: predicted mean (same as imputed if no MC),
              shape ``(N, F)``

            If ``return_uncertainty=True``, additionally:

            - ``"pred_std"``: MC dropout standard deviation, shape ``(N, F)``
            - ``"pred_log_var"``: model-predicted log-variance, shape ``(N, F)``

        Raises:
            RuntimeError: If no graph is available and none is provided.
        """
        # Resolve graph.
        ei = edge_index if edge_index is not None else self._edge_index
        ew = edge_weight if edge_weight is not None else self._edge_weight

        if ei is None:
            raise RuntimeError(
                "No graph available for inference.  Call set_graph() or "
                "load_graph_state() before imputing, or pass edge_index."
            )

        # Prepare tensors.
        if isinstance(features, np.ndarray):
            features_t = torch.from_numpy(features.astype(np.float32))
        else:
            features_t = features.float()

        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask.astype(np.float32))
        else:
            mask_t = mask.float()

        features_t = (features_t * mask_t).to(self.device)
        mask_t = mask_t.to(self.device)
        ei = ei.to(self.device)
        ew = ew.to(self.device) if ew is not None else None

        # Normalize input if scaler is available.
        if self.scaler is not None:
            features_t = self.scaler.transform(features_t, mask_t)

        model_kwargs: dict[str, Any] = {
            "features": features_t,
            "mask": mask_t,
            "edge_index": ei,
            "edge_weight": ew
            if ew is not None
            else torch.ones(ei.shape[1], device=self.device),
            "overlap_frac": torch.ones(ei.shape[1], device=self.device),
            "modality_dims": self.config.modality_dims,
        }

        if not return_uncertainty:
            # Single deterministic forward pass.
            self.model.eval()
            with torch.no_grad():
                output = self.model(**model_kwargs)

            imputed_t = output["imputed"]
            pred_mean_t = output["pred_mean"]
            if self.scaler is not None:
                imputed_t = self.scaler.inverse_transform(imputed_t)
                pred_mean_t = self.scaler.inverse_transform(pred_mean_t)

            return {
                "imputed": imputed_t.cpu().numpy(),
                "pred_mean": pred_mean_t.cpu().numpy(),
            }

        # MC dropout: multiple stochastic forward passes.
        self.model.train()  # Enable dropout.
        mc_predictions: list[np.ndarray] = []
        last_log_var: np.ndarray | None = None

        with torch.no_grad():
            for _ in range(self.mc_samples):
                output = self.model(**model_kwargs)
                imputed_t = output["imputed"]
                if self.scaler is not None:
                    imputed_t = self.scaler.inverse_transform(imputed_t)
                mc_predictions.append(imputed_t.cpu().numpy())
                if "pred_log_var" in output:
                    last_log_var = output["pred_log_var"].cpu().numpy()

        self.model.eval()

        mc_stack = np.stack(mc_predictions, axis=0)  # (S, N, F)
        pred_mean = mc_stack.mean(axis=0)
        pred_std = mc_stack.std(axis=0)

        result: dict[str, np.ndarray] = {
            "imputed": pred_mean,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
        }
        if last_log_var is not None:
            result["pred_log_var"] = last_log_var

        return result

    # ------------------------------------------------------------------
    # DataFrame convenience
    # ------------------------------------------------------------------

    def impute_to_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        return_uncertainty: bool = True,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> pd.DataFrame:
        """Impute from/to a pandas DataFrame.

        Missing values in the DataFrame are identified as NaN.  The
        result is a copy of the input DataFrame with NaN values filled
        by the model's imputed values.  If ``return_uncertainty=True``,
        additional columns ``"<feature>_std"`` are appended.

        Args:
            df: Input DataFrame with patients as rows and features as
                columns.  Must contain all columns listed in
                *feature_columns*.
            feature_columns: Ordered list of feature column names
                matching the model's expected feature order.
            return_uncertainty: Whether to include uncertainty columns.
            edge_index: Optional graph override.
            edge_weight: Optional graph weight override.

        Returns:
            DataFrame with imputed values and optional uncertainty columns.

        Raises:
            ValueError: If required columns are missing from *df*.
        """

        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        # Extract feature matrix and mask.
        feature_matrix = df[feature_columns].values.astype(np.float64)
        mask = (~np.isnan(feature_matrix)).astype(np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0).astype(np.float32)

        # Run imputation.
        result = self.impute(
            features=feature_matrix,
            mask=mask,
            return_uncertainty=return_uncertainty,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        # Build output DataFrame.
        output_df = df.copy()
        imputed = result["imputed"]

        for i, col in enumerate(feature_columns):
            # Only fill missing values; keep observed values unchanged.
            col_mask = mask[:, i].astype(bool)
            output_df.loc[~col_mask, col] = imputed[~col_mask, i]

        # Add uncertainty columns.
        if return_uncertainty and "pred_std" in result:
            pred_std = result["pred_std"]
            for i, col in enumerate(feature_columns):
                output_df[f"{col}_std"] = pred_std[:, i]

        return output_df
