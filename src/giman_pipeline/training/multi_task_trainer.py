"""Multi-task training pipeline for True GIMAN.

Handles training with dynamic task warmup:
- Epochs 1-10: diagnostic only (full labels)
- Epochs 11-30: diagnostic + subtype
- Epochs 31+: all three tasks (survival activated last)

Supports 5-fold CV with per-fold graph reconstruction and MICE imputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

from ..modeling.true_giman import TrueGIMAN, TrueGIMANOutput

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Cox partial likelihood loss (negative log partial likelihood)."""
    sorted_indices = torch.argsort(times, descending=True)
    risk_scores = risk_scores[sorted_indices].squeeze(-1)
    events = events[sorted_indices]

    hazards = torch.exp(risk_scores)
    cumulative_hazards = torch.cumsum(hazards, dim=0)
    log_pl = risk_scores - torch.log(cumulative_hazards + eps)
    log_pl = log_pl * events.float()

    n_events = events.sum()
    if n_events == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    return -log_pl.sum() / (n_events + eps)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for imbalanced classification."""
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()


# ---------------------------------------------------------------------------
# C-index computation
# ---------------------------------------------------------------------------


def concordance_index(
    risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray
) -> float:
    """Compute Harrell's concordance index."""
    n = len(risk_scores)
    concordant = 0
    permissible = 0

    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] > times[i]:
                permissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5

    if permissible == 0:
        return -1.0
    return concordant / permissible


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_knn_graph(
    features: np.ndarray,
    k: int = 10,
    obs_mask: np.ndarray | None = None,
) -> torch.Tensor:
    """Build k-NN graph from feature matrix. Returns edge_index [2, E].

    Args:
        features: [N, F] feature matrix (already imputed/placeholder-filled).
        k: Number of nearest neighbors.
        obs_mask: [N, F] binary mask (1=observed, 0=missing). If provided,
            pairwise distances are computed using only mutually observed features,
            preventing patients with different modality coverage from being
            spuriously connected through imputed/placeholder values.

    Returns:
        edge_index: [2, E] edge index tensor for PyG.
    """
    if obs_mask is None:
        # Standard k-NN (backward compatible)
        adj = kneighbors_graph(
            features, n_neighbors=k, mode="connectivity", include_self=False
        )
        coo = adj.tocoo()
        edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        return edge_index

    # Masked k-NN: pairwise distance using only mutually observed features
    n = features.shape[0]
    mask = obs_mask.astype(np.float32)

    # Compute pairwise squared differences, zeroing out unshared features
    # dist(i, j) = sum_f [ (x_i_f - x_j_f)^2 * mask_i_f * mask_j_f ] / max(overlap, 1)
    # This normalizes by overlap count so patients sharing fewer features
    # aren't automatically "closer" due to fewer terms in the sum.

    # For efficiency, use matrix operations:
    # overlap_ij = mask_i @ mask_j.T  (count of mutually observed features)
    overlap = mask @ mask.T  # [N, N]
    overlap = np.maximum(overlap, 1.0)  # avoid division by zero

    # Masked squared distance: use a loop-free approach
    # sq_diff_sum_ij = sum_f (x_if * m_if - x_jf * m_jf)^2 is not right
    # because we want: sum_f (x_if - x_jf)^2 * m_if * m_jf
    #
    # Expand: sum_f (x_if^2 * m_if * m_jf + x_jf^2 * m_if * m_jf - 2 * x_if * x_jf * m_if * m_jf)
    # = sum_f (x_if^2 * m_if) * m_jf  +  m_if * sum_f (x_jf^2 * m_jf)  - 2 * (x*m)_i @ (x*m)_j^T
    xm = features * mask  # [N, F] — zero out missing features
    x2m = (features**2) * mask  # [N, F]

    # term1: sum_f x_if^2 * m_if * m_jf = x2m @ mask.T  [N, N]
    term1 = x2m @ mask.T
    # term2: sum_f x_jf^2 * m_jf * m_if = mask @ x2m.T = (x2m @ mask.T).T
    term2 = term1.T
    # term3: 2 * (x*m)_i @ (x*m)_j^T
    term3 = 2.0 * (xm @ xm.T)

    dist_sq = (term1 + term2 - term3) / overlap  # [N, N]
    np.fill_diagonal(dist_sq, np.inf)  # exclude self

    # Find k-nearest neighbors per node
    k_actual = min(k, n - 1)
    rows = []
    cols = []
    for i in range(n):
        neighbors = np.argpartition(dist_sq[i], k_actual)[:k_actual]
        rows.extend([i] * k_actual)
        cols.extend(neighbors.tolist())

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiTaskConfig:
    """Training configuration for True GIMAN."""

    # Model
    modality_dims: dict[str, int] = field(default_factory=dict)
    modality_embed_dim: int = 64
    cross_modal_heads: int = 4
    fused_dim: int = 128
    gat_hidden_dim: int = 64
    gat_output_dim: int = 64
    gat_heads: int = 4
    gat_layers: int = 3
    dropout: float = 0.3
    num_subtypes: int = 3
    num_diagnostic_classes: int = 3

    # Training
    lr: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0
    graph_k: int = 10

    # Multi-task warmup schedule (epoch when each task activates)
    warmup_diagnostic: int = 0
    warmup_subtype: int = 10
    warmup_survival: int = 30

    # Task loss weights
    survival_weight: float = 1.0
    subtype_weight: float = 0.5
    diagnostic_weight: float = 0.3

    # Adaptive cross-modal fusion
    adaptive_fusion: bool = False
    observed_threshold: float = 0.0


# ---------------------------------------------------------------------------
# Multi-task trainer
# ---------------------------------------------------------------------------


class MultiTaskTrainer:
    """Trains TrueGIMAN with multi-task warmup scheduling."""

    def __init__(self, config: MultiTaskConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _create_model(self) -> TrueGIMAN:
        cfg = self.config
        return TrueGIMAN(
            modality_dims=cfg.modality_dims,
            modality_embed_dim=cfg.modality_embed_dim,
            cross_modal_heads=cfg.cross_modal_heads,
            fused_dim=cfg.fused_dim,
            gat_hidden_dim=cfg.gat_hidden_dim,
            gat_output_dim=cfg.gat_output_dim,
            gat_heads=cfg.gat_heads,
            gat_layers=cfg.gat_layers,
            num_subtypes=cfg.num_subtypes,
            num_diagnostic_classes=cfg.num_diagnostic_classes,
            dropout=cfg.dropout,
            adaptive_fusion=cfg.adaptive_fusion,
            observed_threshold=cfg.observed_threshold,
        ).to(self.device)

    def _active_tasks(self, epoch: int) -> set[str]:
        """Determine which tasks are active at this epoch."""
        tasks = set()
        cfg = self.config
        if epoch >= cfg.warmup_diagnostic:
            tasks.add("diagnostic")
        if epoch >= cfg.warmup_subtype:
            tasks.add("subtype")
        if epoch >= cfg.warmup_survival:
            tasks.add("survival")
        return tasks

    def _compute_loss(
        self, output: TrueGIMANOutput, data: Data, active_tasks: set[str]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task loss for active tasks."""
        cfg = self.config
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_dict: dict[str, float] = {}

        if "survival" in active_tasks and output.risk_scores is not None:
            # Only compute on patients with survival labels
            mask = data.event >= 0  # All patients have survival data
            if mask.sum() > 0 and data.event[mask].sum() > 0:
                surv_loss = cox_partial_likelihood_loss(
                    output.risk_scores[mask], data.time[mask], data.event[mask]
                )
                total_loss = total_loss + cfg.survival_weight * surv_loss
                loss_dict["survival"] = surv_loss.item()

        if "subtype" in active_tasks and output.subtype_logits is not None:
            if hasattr(data, "subtype") and data.subtype is not None:
                mask = data.subtype >= 0
                if mask.sum() > 0:
                    sub_loss = focal_loss(
                        output.subtype_logits[mask], data.subtype[mask]
                    )
                    total_loss = total_loss + cfg.subtype_weight * sub_loss
                    loss_dict["subtype"] = sub_loss.item()

        if "diagnostic" in active_tasks and output.diagnostic_logits is not None:
            if hasattr(data, "diagnosis") and data.diagnosis is not None:
                mask = data.diagnosis >= 0
                if mask.sum() > 0:
                    diag_loss = focal_loss(
                        output.diagnostic_logits[mask], data.diagnosis[mask]
                    )
                    total_loss = total_loss + cfg.diagnostic_weight * diag_loss
                    loss_dict["diagnostic"] = diag_loss.item()

        return total_loss, loss_dict

    def train_epoch(
        self,
        model: TrueGIMAN,
        data: Data,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch."""
        model.train()
        data = data.to(self.device)
        active_tasks = self._active_tasks(epoch)

        # Pass observation mask if available on the data object
        obs_mask = getattr(data, "obs_mask", None)

        optimizer.zero_grad()
        output = model(data.x, data.edge_index, obs_mask=obs_mask)
        total_loss, loss_dict = self._compute_loss(output, data, active_tasks)

        if total_loss.requires_grad and total_loss.item() > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self.config.grad_clip
            )
            optimizer.step()

        loss_dict["total"] = total_loss.item()
        return loss_dict

    @torch.no_grad()
    def evaluate(self, model: TrueGIMAN, data: Data) -> dict[str, float]:
        """Evaluate model on validation/test data."""
        model.eval()
        data = data.to(self.device)

        # Pass observation mask if available on the data object
        obs_mask = getattr(data, "obs_mask", None)
        output = model(data.x, data.edge_index, obs_mask=obs_mask)
        metrics: dict[str, float] = {}

        # Survival C-index
        if output.risk_scores is not None and hasattr(data, "time"):
            risk_np = output.risk_scores.squeeze(-1).cpu().numpy()
            times_np = data.time.cpu().numpy()
            events_np = data.event.cpu().numpy()
            if events_np.sum() > 0:
                metrics["c_index"] = concordance_index(risk_np, times_np, events_np)

        # Diagnostic AUC
        if (
            output.diagnostic_logits is not None
            and hasattr(data, "diagnosis")
            and data.diagnosis is not None
        ):
            mask = data.diagnosis >= 0
            if mask.sum() > 0:
                preds = output.diagnostic_logits[mask].argmax(dim=1).cpu().numpy()
                targets = data.diagnosis[mask].cpu().numpy()
                metrics["diagnostic_acc"] = (preds == targets).mean()

        # Subtype accuracy
        if (
            output.subtype_logits is not None
            and hasattr(data, "subtype")
            and data.subtype is not None
        ):
            mask = data.subtype >= 0
            if mask.sum() > 0:
                preds = output.subtype_logits[mask].argmax(dim=1).cpu().numpy()
                targets = data.subtype[mask].cpu().numpy()
                metrics["subtype_acc"] = (preds == targets).mean()

        return metrics

    def train_fold(
        self,
        train_data: Data,
        val_data: Data,
        fold: int = 0,
        verbose: bool = True,
    ) -> dict:
        """Train a single fold to completion."""
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
            train_losses = self.train_epoch(model, train_data, optimizer, epoch)
            val_metrics = self.evaluate(model, val_data)

            # Record per-epoch history for training curve visualization
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_losses.get("total", 0.0),
                    "train_survival_loss": train_losses.get("survival", 0.0),
                    "val_c_index": val_metrics.get("c_index", 0.0),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Primary metric for early stopping: C-index if available, else diagnostic accuracy
            primary = val_metrics.get("c_index", val_metrics.get("diagnostic_acc", 0.0))
            scheduler.step(primary)

            if verbose and (epoch + 1) % 20 == 0:
                active = self._active_tasks(epoch)
                loss_str = ", ".join(f"{k}={v:.4f}" for k, v in train_losses.items())
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                print(
                    f"  Fold {fold + 1} Epoch {epoch + 1:3d} [{','.join(active)}]: "
                    f"Loss({loss_str}) Val({metric_str})"
                )

            if primary > best_metric:
                best_metric = primary
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    if verbose:
                        print(f"  Fold {fold + 1}: Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        final_metrics = self.evaluate(model, val_data)
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
        features: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        obs_mask: np.ndarray | None = None,
        diagnosis: np.ndarray | None = None,
        subtypes: np.ndarray | None = None,
        k_folds: int = 5,
    ) -> list[dict]:
        """Run k-fold cross-validation with per-fold graph construction.

        Args:
            features: [N, F] feature matrix (placeholder-filled for missing values).
            times: [N] time-to-event array.
            events: [N] event indicator array.
            obs_mask: [N, F] binary observation mask (1=observed, 0=missing/imputed).
                If None, all features treated as observed (backward compatible).
            diagnosis: [N] optional diagnostic labels.
            subtypes: [N] optional subtype labels.
            k_folds: Number of cross-validation folds.
        """
        print(f"\n{'=' * 60}")
        print(f"TRUE GIMAN: {k_folds}-FOLD CROSS-VALIDATION")
        print(f"{'=' * 60}\n")
        print(
            f"Samples: {len(features)}, Events: {events.sum()}, Event rate: {events.mean():.1%}"
        )
        if obs_mask is not None:
            obs_frac = obs_mask.mean()
            print(f"Observation mask: {obs_frac:.1%} features observed overall")

        n = len(features)
        indices = np.arange(n)

        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, events)):
            print(f"\nFold {fold + 1}/{k_folds}")
            print("-" * 40)

            # Build per-fold graphs (prevents transductive leakage)
            train_edge_index = build_knn_graph(
                features[train_idx], k=self.config.graph_k
            )
            val_edge_index = build_knn_graph(
                features[val_idx], k=min(self.config.graph_k, len(val_idx) - 1)
            )

            # Create PyG Data objects
            train_data = Data(
                x=torch.tensor(features[train_idx], dtype=torch.float32),
                edge_index=train_edge_index,
                time=torch.tensor(times[train_idx], dtype=torch.float32),
                event=torch.tensor(events[train_idx], dtype=torch.long),
            )
            val_data = Data(
                x=torch.tensor(features[val_idx], dtype=torch.float32),
                edge_index=val_edge_index,
                time=torch.tensor(times[val_idx], dtype=torch.float32),
                event=torch.tensor(events[val_idx], dtype=torch.long),
            )

            # Add observation masks
            if obs_mask is not None:
                train_data.obs_mask = torch.tensor(
                    obs_mask[train_idx], dtype=torch.float32
                )
                val_data.obs_mask = torch.tensor(obs_mask[val_idx], dtype=torch.float32)

            # Add optional labels
            if diagnosis is not None:
                train_data.diagnosis = torch.tensor(
                    diagnosis[train_idx], dtype=torch.long
                )
                val_data.diagnosis = torch.tensor(diagnosis[val_idx], dtype=torch.long)
            if subtypes is not None:
                train_data.subtype = torch.tensor(subtypes[train_idx], dtype=torch.long)
                val_data.subtype = torch.tensor(subtypes[val_idx], dtype=torch.long)

            print(
                f"  Train: {len(train_idx)} samples, {events[train_idx].sum()} events"
            )
            print(f"  Val: {len(val_idx)} samples, {events[val_idx].sum()} events")

            result = self.train_fold(train_data, val_data, fold=fold)
            fold_results.append(result)

            metrics_str = ", ".join(
                f"{k}={v:.4f}" for k, v in result["final_metrics"].items()
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

        for metric, values in all_metrics.items():
            arr = np.array(values)
            print(f"  {metric}: {arr.mean():.4f} +/- {arr.std():.4f}")

        return fold_results
