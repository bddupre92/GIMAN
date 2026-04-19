"""PhysGIMINTrainer — training loop with LR-annealing λ scheduler + recon-floor clamp.

Implements the Wang-Teng-Perdikaris 2021 gradient-ratio EMA scheduler for λ_phys,
with a hard recon-floor clamp (min_recon_fraction=0.30) to prevent σ collapse.

Robustness contract (all 6 layers):
  Layer 1 — Seed-first: explicit seed, torch.use_deterministic_algorithms(warn_only=True)
  Layer 2 — Per-seed output dir: {output_dir}/{run_name}_seed{N}_{timestamp}/
  Layer 3 — Atomic status writes: .status.json.tmp + replace
  Layer 3b — NaN handling: fail-fast with status='failed', reason='nan_loss'|'nan_grad'
  Layer 4 — checkpoint.pt saves model.state_dict() only (bytes-stable across seeds)
  Layer 5 — config.json + provenance.json at startup; training_history.json per epoch
  Layer 6 — EarlyStopping on val_loss with patience

Reference: impl_best_practices.md §2-§3 (β-NLL + stop-grad + LR-annealing recipe).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from phys_gimin.loss import physgimin_loss, PhysGIMINLossComponents
from phys_gimin.model import PhysGIMIN
from phys_gimin.regularizer import PhysicsRegularizer
from phys_gimin.trajectory_cache import TrajectoryCache


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class EpochRecord:
    """Per-epoch training record. Serialised to training_history.json."""
    epoch: int
    train_loss: float
    val_loss: float
    l_recon: float
    l_physics: float
    lambda_phys: float
    recon_fraction: float
    floor_violation: bool
    grad_norm_recon: float
    grad_norm_physics: float
    wall_time_s: float


@dataclass
class TrainingResult:
    """Summary returned from PhysGIMINTrainer.fit()."""
    status: str          # "completed" | "failed"
    final_epoch: int
    best_val_loss: float
    wall_time_s: float
    run_dir: str
    history: list[EpochRecord]
    failure_reason: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _git_sha(cwd: Path | None = None) -> str:
    """Return current git HEAD SHA or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _grad_l2_norm(loss: Tensor, model: PhysGIMIN) -> float:
    """Compute L2 norm of gradients of `loss` w.r.t. model parameters.

    Uses torch.autograd.grad with retain_graph=True and allow_unused=True so
    that this can be called without consuming the graph. Caller must zero_grad()
    and call the actual backward() separately.
    """
    grads = torch.autograd.grad(
        loss,
        list(model.parameters()),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total_sq = sum(
        g.pow(2).sum().item()
        for g in grads
        if g is not None
    )
    return math.sqrt(total_sq) if total_sq > 0.0 else 0.0


# ── Trainer ──────────────────────────────────────────────────────────────────

class PhysGIMINTrainer:
    """Training loop for PhysGIMIN with LR-annealing λ + recon-floor clamp + robustness.

    Algorithm (Wang-Teng-Perdikaris 2021, §3.3):
    1. Forward pass → mu, log_var.
    2. L_recon = beta_nll(mu_all, sigma_all, target).
    3. L_physics = PhysicsRegularizer(mu_ode, sigma_ode.detach(), ...).
    4. L_total = L_recon + lambda_phys * L_physics.
    5. Grad-norm isolation (3 backward passes per step):
         a) L_recon.backward(retain_graph=True) → collect grads → zero_grad()
         b) L_physics.backward(retain_graph=True) → collect grads → zero_grad()
         c) L_total.backward() → optimizer.step()
    6. EMA λ update: lambda_phys ← alpha * lambda_phys + (1-alpha) * (||∇L_recon|| / ||∇L_phys||)
    7. Warmup: lambda_phys = 0 for epochs < warmup_epochs.
    8. Clamp: lambda_phys = min(lambda_phys, lambda_phys_target).
    9. Recon-floor clamp: if L_recon / L_total < min_recon_fraction → lambda_phys *= 0.5.
    """

    def __init__(
        self,
        model: PhysGIMIN,
        regularizer: PhysicsRegularizer,
        trajectory_cache: TrajectoryCache,
        optimizer: torch.optim.Optimizer,
        seed: int,
        lambda_phys_target: float = 1.0,
        warmup_epochs: int = 5,
        ema_alpha: float = 0.9,
        min_recon_fraction: float = 0.30,
        floor_violation_decay: float = 0.5,
        output_dir: Path = Path("outputs/paper12_phys_gimin/runs"),
        run_name: str = "phys_gimin_default",
        deterministic: bool = True,
        device: str = "cpu",
    ) -> None:
        # ── Layer 1: Seed-first ───────────────────────────────────────────────
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.model = model
        self.regularizer = regularizer
        self.trajectory_cache = trajectory_cache
        self.optimizer = optimizer
        self.seed = seed
        self.lambda_phys = 0.0          # current λ — starts at 0 (warmup)
        self.lambda_phys_target = float(lambda_phys_target)
        self.warmup_epochs = warmup_epochs
        self.ema_alpha = float(ema_alpha)
        self.min_recon_fraction = float(min_recon_fraction)
        self.floor_violation_decay = float(floor_violation_decay)
        self.device = torch.device(device)
        self.floor_violation_count = 0

        # ── Layer 2: Per-seed output directory ───────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"{run_name}_seed{seed}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Save initial model state so fit() can restore it for deterministic replay
        import io as _io
        _buf = _io.BytesIO()
        torch.save(self.model.state_dict(), _buf)
        self._initial_model_state = _buf.getvalue()

        # ── Write startup files ───────────────────────────────────────────────
        self._write_config_json()
        self._write_provenance_json()
        self._write_status_atomic({
            "status": "started",
            "pid": os.getpid(),
            "start_ts": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "run_dir": str(self.run_dir),
        })

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        patience: int = 20,
    ) -> TrainingResult:
        """Run training loop. Writes all JSONs to self.run_dir. Returns TrainingResult."""
        start_time = datetime.now()
        history: list[EpochRecord] = []
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        # Re-seed at fit() entry AND restore initial model weights to ensure
        # determinism regardless of what happened between __init__ and fit()
        # (e.g., other RNG consumers, dataset creation). Both are needed:
        #   - torch.manual_seed resets the RNG stream for dropout etc.
        #   - load_state_dict restores weights to the post-__init__ state
        import io as _io
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        _state = torch.load(_io.BytesIO(self._initial_model_state), weights_only=True)
        self.model.load_state_dict(_state)

        try:
            for epoch in range(n_epochs):
                epoch_start = datetime.now()

                # Flush trajectory cache once per epoch (amortized cost)
                self.trajectory_cache.advance_epoch()

                # ── Training pass ─────────────────────────────────────────────
                self.model.train()
                train_records: list[PhysGIMINLossComponents] = []
                train_gnorm_recon = 0.0
                train_gnorm_physics = 0.0

                for batch in train_loader:
                    batch = _move_batch_to_device(batch, self.device)
                    components = self._step(batch, epoch=epoch)
                    train_records.append(components)
                    # Accumulate grad norms from last step (stored on self)
                    train_gnorm_recon += self._last_gnorm_recon
                    train_gnorm_physics += self._last_gnorm_physics

                n_batches = max(1, len(train_records))
                avg_train_loss = sum(c.total.item() if isinstance(c.total, Tensor) else float(c.total) for c in train_records) / n_batches
                avg_recon = sum(c.recon.item() if isinstance(c.recon, Tensor) else float(c.recon) for c in train_records) / n_batches
                avg_physics = sum(c.physics.item() if isinstance(c.physics, Tensor) else float(c.physics) for c in train_records) / n_batches
                avg_recon_frac = sum(c.recon_fraction for c in train_records) / n_batches
                floor_violation = any(c.recon_fraction < self.min_recon_fraction for c in train_records)

                # ── Validation pass ───────────────────────────────────────────
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = _move_batch_to_device(batch, self.device)
                        val_components = self._val_step(batch)
                        val_losses.append(float(val_components.total.item() if isinstance(val_components.total, Tensor) else val_components.total))

                avg_val_loss = sum(val_losses) / max(1, len(val_losses))

                # ── Epoch timing ──────────────────────────────────────────────
                wall_time_s = (datetime.now() - epoch_start).total_seconds()

                record = EpochRecord(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    l_recon=avg_recon,
                    l_physics=avg_physics,
                    lambda_phys=self.lambda_phys,
                    recon_fraction=avg_recon_frac,
                    floor_violation=floor_violation,
                    grad_norm_recon=train_gnorm_recon / n_batches,
                    grad_norm_physics=train_gnorm_physics / n_batches,
                    wall_time_s=wall_time_s,
                )
                history.append(record)

                # ── Early stopping ────────────────────────────────────────────
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    # Save best checkpoint
                    torch.save(self.model.state_dict(), self.run_dir / "checkpoint.pt")
                else:
                    epochs_no_improve += 1

                # ── Atomic status update per epoch ────────────────────────────
                self._write_status_atomic({
                    "status": "running",
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "lambda_phys": self.lambda_phys,
                    "floor_violation_count": self.floor_violation_count,
                })

                if epochs_no_improve >= patience:
                    break

            # ── Finalization ──────────────────────────────────────────────────
            # Save checkpoint if not saved yet (no improvement ever)
            if not (self.run_dir / "checkpoint.pt").exists():
                torch.save(self.model.state_dict(), self.run_dir / "checkpoint.pt")

            wall_time_total = (datetime.now() - start_time).total_seconds()
            self._write_training_history(history)
            self._write_status_atomic({
                "status": "completed",
                "final_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "wall_time_s": wall_time_total,
                "floor_violation_count": self.floor_violation_count,
            })

            return TrainingResult(
                status="completed",
                final_epoch=best_epoch,
                best_val_loss=best_val_loss,
                wall_time_s=wall_time_total,
                run_dir=str(self.run_dir),
                history=history,
                failure_reason=None,
            )

        except Exception as exc:
            wall_time_total = (datetime.now() - start_time).total_seconds()
            tb_str = traceback.format_exc()
            reason = getattr(exc, "_phys_reason", str(type(exc).__name__))

            self._write_status_atomic({
                "status": "failed",
                "reason": reason,
                "traceback": tb_str,
                "wall_time_s": wall_time_total,
                "floor_violation_count": self.floor_violation_count,
            })
            self._write_training_history(history)

            return TrainingResult(
                status="failed",
                final_epoch=len(history) - 1 if history else 0,
                best_val_loss=best_val_loss,
                wall_time_s=wall_time_total,
                run_dir=str(self.run_dir),
                history=history,
                failure_reason=reason,
            )

    # ── Private: single training step ─────────────────────────────────────────

    def _step(self, batch: dict[str, Any], epoch: int = 0) -> PhysGIMINLossComponents:
        """Single training step with grad-norm isolation (3 backward passes).

        Pass order:
            1. L_recon.backward(retain_graph=True) → grad_norm_recon, then zero_grad
            2. L_physics.backward(retain_graph=True) → grad_norm_physics, then zero_grad
            3. L_total.backward() → optimizer.step()

        The three-backward-pass approach is intentional per impl_best_practices.md §2.
        It isolates grad norms per loss term for the EMA scheduler without materialising
        second-order derivatives (create_graph=False).
        """
        self.optimizer.zero_grad()

        # ── Forward pass ──────────────────────────────────────────────────────
        features = batch["features"]
        mask = batch["mask"]
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        overlap_frac = batch["overlap_frac"]
        stage_ids = batch["stage_ids"]
        patnos = batch["patnos"]
        t_years_per_patient = batch["t_years_per_patient"]
        sbr_0_per_patient = batch["sbr_0_per_patient"]
        ode_feature_indices = batch["ode_feature_indices"]

        output = self.model(
            features,
            mask,
            edge_index,
            edge_weight,
            overlap_frac,
            stage_ids,
        )

        mu_all = output["imputed_mean"]               # (N, F)
        log_var_all = output["imputed_log_var"]        # (N, F)
        sigma_all = torch.exp(0.5 * log_var_all).clamp(min=1e-3)

        # Reconstruction targets: observed values (masked features)
        target_recon = features

        # ODE subset: select physics-relevant features
        mu_ode = mu_all[:, ode_feature_indices]        # (N, n_ode)
        sigma_ode = sigma_all[:, ode_feature_indices]  # (N, n_ode)

        # Compute loss components (physgimin_loss handles the stop-grad on sigma_ode)
        # lambda_phys is 0 during warmup — physics term contributes nothing to total
        current_lambda = self.lambda_phys if epoch >= self.warmup_epochs else 0.0

        components = physgimin_loss(
            mu_recon=mu_all,
            sigma_recon=sigma_all,
            target_recon=target_recon,
            mu_ode=mu_ode,
            sigma_ode=sigma_ode,
            regularizer=self.regularizer,
            patnos=patnos,
            t_years_per_patient=t_years_per_patient,
            sbr_0_per_patient=sbr_0_per_patient,
            lambda_phys=current_lambda,
            beta=0.5,
            min_recon_fraction=self.min_recon_fraction,
        )

        l_recon = components.recon
        l_physics = components.physics
        l_total = components.total

        # ── NaN check (Layer 3b) ──────────────────────────────────────────────
        if torch.isnan(l_total):
            exc = RuntimeError("NaN loss detected during training.")
            exc._phys_reason = "nan_loss"
            raise exc

        # ── Grad-norm isolation: pass 1 (recon) ──────────────────────────────
        # Re-compute recon loss with grad to get grad norms. We use autograd.grad
        # rather than calling .backward() directly so we can inspect per-term norms
        # without disturbing the main computation graph.
        try:
            gnorm_recon = _grad_l2_norm(
                l_recon if l_recon.requires_grad else l_total,
                self.model,
            )
        except Exception:
            gnorm_recon = 0.0

        # ── Grad-norm isolation: pass 2 (physics) ────────────────────────────
        try:
            if current_lambda > 0.0 and l_physics.requires_grad:
                gnorm_physics = _grad_l2_norm(l_physics, self.model)
            else:
                gnorm_physics = 0.0
        except Exception:
            gnorm_physics = 0.0

        # Store on self for accumulation in fit()
        self._last_gnorm_recon = gnorm_recon
        self._last_gnorm_physics = gnorm_physics

        # ── EMA λ scheduler update (per Wang 2021) ────────────────────────────
        if epoch >= self.warmup_epochs:
            eps = 1e-8
            if gnorm_physics > eps and math.isfinite(gnorm_recon) and math.isfinite(gnorm_physics):
                lambda_hat = gnorm_recon / (gnorm_physics + eps)
                self.lambda_phys = (
                    self.ema_alpha * self.lambda_phys
                    + (1.0 - self.ema_alpha) * lambda_hat
                )
            # Clamp to target ceiling
            self.lambda_phys = min(self.lambda_phys, self.lambda_phys_target)

            # Recon-floor clamp (Anti-Pattern 1 defense)
            recon_frac = components.recon_fraction
            if recon_frac < self.min_recon_fraction:
                self.lambda_phys *= self.floor_violation_decay
                self.floor_violation_count += 1

        # ── Pass 3: actual backward + optimizer step ──────────────────────────
        self.optimizer.zero_grad()
        l_total.backward()

        # NaN gradient check (Layer 3b)
        for p in self.model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                exc = RuntimeError("NaN gradient detected during training.")
                exc._phys_reason = "nan_grad"
                raise exc

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return components

    def _val_step(self, batch: dict[str, Any]) -> PhysGIMINLossComponents:
        """Forward-only val step (no backward)."""
        features = batch["features"]
        mask = batch["mask"]
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        overlap_frac = batch["overlap_frac"]
        stage_ids = batch["stage_ids"]
        patnos = batch["patnos"]
        t_years_per_patient = batch["t_years_per_patient"]
        sbr_0_per_patient = batch["sbr_0_per_patient"]
        ode_feature_indices = batch["ode_feature_indices"]

        output = self.model(
            features, mask, edge_index, edge_weight, overlap_frac, stage_ids,
        )
        mu_all = output["imputed_mean"]
        log_var_all = output["imputed_log_var"]
        sigma_all = torch.exp(0.5 * log_var_all).clamp(min=1e-3)
        mu_ode = mu_all[:, ode_feature_indices]
        sigma_ode = sigma_all[:, ode_feature_indices]

        return physgimin_loss(
            mu_recon=mu_all,
            sigma_recon=sigma_all,
            target_recon=features,
            mu_ode=mu_ode,
            sigma_ode=sigma_ode,
            regularizer=self.regularizer,
            patnos=patnos,
            t_years_per_patient=t_years_per_patient,
            sbr_0_per_patient=sbr_0_per_patient,
            lambda_phys=self.lambda_phys,
            beta=0.5,
            min_recon_fraction=self.min_recon_fraction,
        )

    # ── Private: JSON writers ─────────────────────────────────────────────────

    def _write_status_atomic(self, status_dict: dict) -> None:
        """Atomic write of status.json via tmp + replace (POSIX atomic)."""
        tmp_path = self.run_dir / ".status.json.tmp"
        tmp_path.write_text(json.dumps(status_dict, indent=2))
        tmp_path.replace(self.run_dir / "status.json")

    def _write_config_json(self) -> None:
        """Write config.json with constructor hyperparameters + git SHA + model class."""
        config = {
            "run_name": str(self.run_dir.name),
            "seed": self.seed,
            "lambda_phys_target": self.lambda_phys_target,
            "warmup_epochs": self.warmup_epochs,
            "ema_alpha": self.ema_alpha,
            "min_recon_fraction": self.min_recon_fraction,
            "floor_violation_decay": self.floor_violation_decay,
            "device": str(self.device),
            "model_class": type(self.model).__name__,
            "git_sha": _git_sha(cwd=Path(__file__).parent),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def _write_provenance_json(self) -> None:
        """Write provenance.json with prior_source_hash + variant_label + git SHA."""
        provenance = {
            "prior_source_hash": self.regularizer.prior_source_hash,
            "variant_label": self.regularizer.variant_label,
            "git_sha": _git_sha(cwd=Path(__file__).parent),
            "model_class": type(self.model).__name__,
            "regularizer_class": type(self.regularizer).__name__,
            "provider_class": type(self.regularizer.provider).__name__,
        }
        (self.run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    def _write_training_history(self, history: list[EpochRecord]) -> None:
        """Write training_history.json — list of EpochRecord dicts."""
        records = [asdict(r) for r in history]
        (self.run_dir / "training_history.json").write_text(json.dumps(records, indent=2))


# ── Utility ───────────────────────────────────────────────────────────────────

def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields in a batch dict to the given device. Non-tensors left as-is."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
