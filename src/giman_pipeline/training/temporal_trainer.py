"""Temporal multi-task training pipeline for Temporal GIMAN.

Adapts the MultiTaskTrainer for longitudinal temporal sequences:
- Mini-batch DataLoader instead of full-graph single pass
- Graph construction per batch from last hidden states
- Cox loss on final-step risk + trajectory consistency loss
- Dynamic C-index at landmark times (12, 24, 36, 48 months)
- 5-fold stratified CV with temporal collation

Reuses loss functions and metrics from multi_task_trainer.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from ..modeling.temporal_giman import TemporalGIMAN, TemporalGIMANOutput
from .multi_task_trainer import concordance_index, cox_partial_likelihood_loss
from .temporal_data_loaders import collate_temporal_fn


# ---------------------------------------------------------------------------
# Dynamic C-index
# ---------------------------------------------------------------------------


def dynamic_concordance_index(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    landmark_time: float,
) -> float:
    """C-index restricted to patients at risk at a given landmark time.

    Only evaluates concordance among patients whose follow-up extends
    at least to landmark_time, providing time-dependent discrimination.

    Args:
        risk_scores: [N] predicted risk scores.
        times: [N] time-to-event values.
        events: [N] event indicators.
        landmark_time: Time point for evaluation (in months).

    Returns:
        Dynamic C-index, or -1.0 if insufficient subjects.
    """
    at_risk = times >= landmark_time
    if at_risk.sum() < 10 or events[at_risk].sum() == 0:
        return -1.0
    return concordance_index(
        risk_scores[at_risk], times[at_risk], events[at_risk]
    )


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TemporalTrainingConfig:
    """Training configuration for Temporal GIMAN."""

    # Model architecture
    modality_dims: dict[str, int] = field(default_factory=dict)
    modality_embed_dim: int = 64
    cross_modal_heads: int = 4
    fused_dim: int = 128
    gat_hidden_dim: int = 64
    gat_output_dim: int = 64
    gat_heads: int = 4
    gat_layers: int = 3
    dropout: float = 0.3
    adaptive_fusion: bool = True
    observed_threshold: float = 0.3

    # Temporal encoder
    temporal_hidden_dim: int = 64
    temporal_num_layers: int = 2

    # Training
    lr: float = 0.0005
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 25
    grad_clip: float = 1.0
    batch_size: int = 64

    # Graph
    graph_k: int = 10

    # Loss weights
    cox_loss_weight: float = 1.0
    trajectory_loss_weight: float = 0.5

    # Evaluation
    landmark_times: list[int] = field(
        default_factory=lambda: [12, 24, 36, 48]
    )


# ---------------------------------------------------------------------------
# Temporal multi-task trainer
# ---------------------------------------------------------------------------


class TemporalMultiTaskTrainer:
    """Trains TemporalGIMAN with mini-batch DataLoader and temporal losses."""

    def __init__(
        self,
        config: TemporalTrainingConfig,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _create_model(self) -> TemporalGIMAN:
        cfg = self.config
        return TemporalGIMAN(
            modality_dims=cfg.modality_dims,
            modality_embed_dim=cfg.modality_embed_dim,
            cross_modal_heads=cfg.cross_modal_heads,
            fused_dim=cfg.fused_dim,
            temporal_hidden_dim=cfg.temporal_hidden_dim,
            temporal_num_layers=cfg.temporal_num_layers,
            gat_hidden_dim=cfg.gat_hidden_dim,
            gat_output_dim=cfg.gat_output_dim,
            gat_heads=cfg.gat_heads,
            gat_layers=cfg.gat_layers,
            dropout=cfg.dropout,
            adaptive_fusion=cfg.adaptive_fusion,
            observed_threshold=cfg.observed_threshold,
            graph_k=cfg.graph_k,
        ).to(self.device)

    @staticmethod
    def _extract_last_valid(
        trajectories: torch.Tensor,
        n_visits: torch.Tensor,
    ) -> torch.Tensor:
        """Extract last valid value from trajectory tensor.

        Args:
            trajectories: [N, T, 1] trajectory values.
            n_visits: [N] number of valid visits.

        Returns:
            last_values: [N, 1] last valid trajectory values.
        """
        N = trajectories.size(0)
        last_idx = (n_visits - 1).clamp(min=0).long()
        last_idx_expanded = last_idx.unsqueeze(1).unsqueeze(2)
        last_values = trajectories.gather(1, last_idx_expanded).squeeze(1)
        return last_values

    def _compute_temporal_loss(
        self,
        output: TemporalGIMANOutput,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute temporal multi-task loss.

        Components:
        1. Cox partial likelihood on final-step risk scores
        2. Trajectory consistency: endpoint of monotone trajectory
           should agree with final-step risk score

        Args:
            output: Model output from forward pass.
            batch: Collated batch dict from DataLoader.

        Returns:
            total_loss: Scalar loss tensor.
            loss_dict: Per-component loss values for logging.
        """
        cfg = self.config
        total_loss = torch.tensor(
            0.0, device=self.device, requires_grad=True
        )
        loss_dict: dict[str, float] = {}

        time_to_event = batch["time_to_event"].to(self.device)
        event = batch["event"].to(self.device)

        # Cox loss on final-step risk
        if output.risk_scores is not None and event.sum() > 0:
            cox_loss = cox_partial_likelihood_loss(
                output.risk_scores, time_to_event, event
            )
            total_loss = total_loss + cfg.cox_loss_weight * cox_loss
            loss_dict["cox"] = cox_loss.item()

        # Trajectory consistency loss
        if (
            output.risk_trajectories is not None
            and output.risk_scores is not None
            and cfg.trajectory_loss_weight > 0
        ):
            n_visits = batch["n_visits"].to(self.device)
            last_traj = self._extract_last_valid(
                output.risk_trajectories, n_visits
            )
            # MSE between trajectory endpoint and final risk score
            # Detach risk_scores so gradient flows through trajectory head
            traj_loss = F.mse_loss(last_traj, output.risk_scores.detach())
            total_loss = total_loss + cfg.trajectory_loss_weight * traj_loss
            loss_dict["trajectory"] = traj_loss.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def _batch_to_device(
        self, batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device)
            else:
                device_batch[k] = v
        return device_batch

    def train_epoch(
        self,
        model: TemporalGIMAN,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Train for one epoch over mini-batches."""
        model.train()
        epoch_losses: dict[str, list[float]] = {}

        for batch in dataloader:
            batch = self._batch_to_device(batch)
            optimizer.zero_grad()

            output = model(
                features=batch["features"],
                obs_mask=batch["obs_mask"],
                time_months=batch["time_months"],
                seq_mask=batch["seq_mask"],
                n_visits=batch["n_visits"],
            )

            total_loss, loss_dict = self._compute_temporal_loss(output, batch)

            if total_loss.requires_grad and total_loss.item() > 0:
                total_loss.backward()
                clip_grad_norm_(
                    model.parameters(), max_norm=self.config.grad_clip
                )
                optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses.setdefault(k, []).append(v)

        # Average losses across batches
        return {k: np.mean(v) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(
        self,
        model: TemporalGIMAN,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate model on validation data.

        Computes:
        - Standard C-index on final-step risk
        - Dynamic C-index at each landmark time
        - Monotonicity rate of predicted trajectories
        """
        model.eval()
        all_risk = []
        all_times = []
        all_events = []
        all_trajectories = []
        all_n_visits = []

        for batch in dataloader:
            batch = self._batch_to_device(batch)
            output = model(
                features=batch["features"],
                obs_mask=batch["obs_mask"],
                time_months=batch["time_months"],
                seq_mask=batch["seq_mask"],
                n_visits=batch["n_visits"],
            )

            if output.risk_scores is not None:
                all_risk.append(
                    output.risk_scores.squeeze(-1).cpu().numpy()
                )
            all_times.append(batch["time_to_event"].cpu().numpy())
            all_events.append(batch["event"].cpu().numpy())
            if output.risk_trajectories is not None:
                all_trajectories.append(
                    output.risk_trajectories.cpu().numpy()
                )
                all_n_visits.append(batch["n_visits"].cpu().numpy())

        metrics: dict[str, float] = {}

        if not all_risk:
            return metrics

        risk_np = np.concatenate(all_risk)
        times_np = np.concatenate(all_times)
        events_np = np.concatenate(all_events)

        # Standard C-index
        if events_np.sum() > 0:
            metrics["c_index"] = concordance_index(
                risk_np, times_np, events_np
            )

        # Dynamic C-index at landmark times
        for landmark in self.config.landmark_times:
            dc = dynamic_concordance_index(
                risk_np, times_np, events_np, float(landmark)
            )
            if dc >= 0:
                metrics[f"dc_index_{landmark}m"] = dc

        # Monotonicity check on trajectories
        # Process per-batch to handle variable T_max across batches
        if all_trajectories:
            monotone_count = 0
            total_count = 0
            for traj_batch, visits_batch in zip(
                all_trajectories, all_n_visits
            ):
                for i in range(len(traj_batch)):
                    t = int(visits_batch[i])
                    if t < 2:
                        continue
                    patient_traj = traj_batch[i, :t, 0]
                    diffs = np.diff(patient_traj)
                    is_monotone = np.all(diffs >= -1e-6)
                    monotone_count += int(is_monotone)
                    total_count += 1
            if total_count > 0:
                metrics["monotonicity_rate"] = monotone_count / total_count

        return metrics

    def train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold: int = 0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Train a single fold to completion with early stopping."""
        cfg = self.config
        model = self._create_model()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        best_metric = -float("inf")
        best_state = None
        patience_counter = 0
        training_history: list[dict] = []

        for epoch in range(cfg.max_epochs):
            train_losses = self.train_epoch(model, train_loader, optimizer)
            val_metrics = self.evaluate(model, val_loader)

            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_losses.get("total", 0.0),
                    "train_cox_loss": train_losses.get("cox", 0.0),
                    "train_traj_loss": train_losses.get("trajectory", 0.0),
                    "val_c_index": val_metrics.get("c_index", 0.0),
                    "val_dc_24m": val_metrics.get("dc_index_24m", 0.0),
                    "val_monotonicity": val_metrics.get(
                        "monotonicity_rate", 0.0
                    ),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Primary metric: dynamic C-index at 24 months, fallback to standard
            primary = val_metrics.get(
                "dc_index_24m", val_metrics.get("c_index", 0.0)
            )
            scheduler.step(primary)

            if verbose and (epoch + 1) % 20 == 0:
                loss_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in train_losses.items()
                )
                metric_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                )
                print(
                    f"  Fold {fold + 1} Epoch {epoch + 1:3d}: "
                    f"Loss({loss_str}) Val({metric_str})"
                )

            if primary > best_metric:
                best_metric = primary
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    if verbose:
                        print(
                            f"  Fold {fold + 1}: Early stopping "
                            f"at epoch {epoch + 1}"
                        )
                    break

        # Load best model for final evaluation
        if best_state is not None:
            model.load_state_dict(best_state)

        final_metrics = self.evaluate(model, val_loader)

        return {
            "fold": fold + 1,
            "best_primary_metric": best_metric,
            "final_metrics": final_metrics,
            "model_state": best_state,
            "epochs_trained": epoch + 1,
            "training_history": training_history,
        }

    def cross_validate(
        self,
        dataset: Any,
        k_folds: int = 5,
    ) -> list[dict]:
        """Run k-fold cross-validation.

        Args:
            dataset: TemporalPPMIDataset instance.
            k_folds: Number of CV folds.

        Returns:
            List of per-fold result dicts.
        """
        cfg = self.config

        print(f"\n{'=' * 60}")
        print(f"TEMPORAL GIMAN: {k_folds}-FOLD CROSS-VALIDATION")
        print(f"{'=' * 60}\n")
        print(f"Patients: {len(dataset)}")
        print(f"Device: {self.device}")
        print(f"Batch size: {cfg.batch_size}")
        print(f"Landmark times: {cfg.landmark_times}")

        # Extract event indicators for stratification
        events = np.array(
            [dataset[i]["event"].item() for i in range(len(dataset))]
        )
        indices = np.arange(len(dataset))

        print(
            f"Events: {events.sum():.0f}/{len(events)} "
            f"({events.mean():.1%})"
        )

        kfold = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=42
        )
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            kfold.split(indices, events.astype(int))
        ):
            print(f"\nFold {fold + 1}/{k_folds}")
            print("-" * 40)
            print(
                f"  Train: {len(train_idx)} patients, "
                f"{events[train_idx].sum():.0f} events"
            )
            print(
                f"  Val: {len(val_idx)} patients, "
                f"{events[val_idx].sum():.0f} events"
            )

            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())

            train_loader = DataLoader(
                train_subset,
                batch_size=cfg.batch_size,
                collate_fn=collate_temporal_fn,
                shuffle=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=cfg.batch_size,
                collate_fn=collate_temporal_fn,
                shuffle=False,
                drop_last=False,
            )

            result = self.train_fold(train_loader, val_loader, fold=fold)
            fold_results.append(result)

            metrics_str = ", ".join(
                f"{k}={v:.4f}"
                for k, v in result["final_metrics"].items()
            )
            print(f"  Fold {fold + 1} final: {metrics_str}")

        # Summary
        print(f"\n{'=' * 60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'=' * 60}")

        all_metrics: dict[str, list[float]] = {}
        for r in fold_results:
            for k, v in r["final_metrics"].items():
                all_metrics.setdefault(k, []).append(v)

        for metric, values in sorted(all_metrics.items()):
            arr = np.array(values)
            print(f"  {metric}: {arr.mean():.4f} +/- {arr.std():.4f}")

        return fold_results
