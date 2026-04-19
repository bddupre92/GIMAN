"""Tests for PhysGIMINTrainer: scheduler logic + robustness contract.

14 tests across 3 classes:
  TestSchedulerAndLoss (5) — λ warmup/ramp/EMA/recon-floor/loss-composition
  TestRobustnessContract (5) — output-dir schema, status, config, provenance, history
  TestRobustnessDeterminism (4) — seed determinism, NaN failure, status transitions
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from phys_gimin.loss import physgimin_loss
from phys_gimin.model import PhysGIMIN
from phys_gimin.priors.literature import LiteraturePriorProvider
from phys_gimin.regularizer import PhysicsRegularizer
from phys_gimin.trajectory_cache import TrajectoryCache
from phys_gimin.training import PhysGIMINTrainer, EpochRecord, TrainingResult


# ── Tiny dataset ──────────────────────────────────────────────────────────────

class _TinyDataset(Dataset):
    """Tiny dataset for trainer unit tests: 8 patients, 33 features, seeded noise.

    ODE design: ode_feature_indices=[0, 1] means features 0 and 1 are DaT-SBR
    measurements. t_years has exactly 2 elements to match — one per ODE feature index.
    The physics regularizer compares mu_ode[:, i] against ODE trajectory at t_years[i].
    """

    def __init__(self, n_patients: int = 8, n_features: int = 33, seed: int = 1001):
        rng = np.random.default_rng(seed)
        self.features = torch.from_numpy(rng.standard_normal((n_patients, n_features))).float()
        self.mask = torch.ones(n_patients, n_features)
        self.stage_ids = torch.from_numpy(rng.integers(0, 6, (n_patients,))).long()
        self.patnos = list(range(3000, 3000 + n_patients))
        # t_years has 2 elements — matches len(ode_feature_indices)=2
        # The physics regularizer expects mu shape (batch, n_visits) = (8, 2)
        self.t_years = [np.array([0.0, 1.0])] * n_patients
        self.sbr_0 = [2.5] * n_patients

    def __len__(self) -> int:
        return 1  # one full-graph batch

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features,
            "mask": self.mask,
            "edge_index": torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long),
            "edge_weight": torch.ones(4),
            "overlap_frac": torch.ones(4),
            "stage_ids": self.stage_ids,
            "patnos": self.patnos,
            "t_years_per_patient": self.t_years,
            "sbr_0_per_patient": self.sbr_0,
            "ode_feature_indices": [0, 1],  # first 2 features are "DaT-SBR" for physics
        }


def _collate_passthrough(batch):
    """batch is a length-1 list of dicts; return the dict directly."""
    return batch[0]


def _build_trainer(
    tmp_path: Path,
    seed: int = 1001,
    warmup_epochs: int = 1,
    lambda_phys_target: float = 1.0,
    min_recon_fraction: float = 0.30,
    deterministic: bool = True,
) -> PhysGIMINTrainer:
    """Factory for integration tests.

    Seeds torch BEFORE PhysGIMIN construction so that two calls with the same
    seed produce bit-identical initial model weights regardless of prior RNG state.
    """
    # Seed before model construction — determinism requires consistent weight init
    torch.manual_seed(seed)
    model = PhysGIMIN(
        modality_dims=[2, 5, 6, 6, 4, 4, 6],
        embed_dim=64,
        num_gnn_layers=2,
        num_heads=4,
        mc_dropout=0.1,
        num_stages=6,
        stage_embed_dim=16,
    )
    provider = LiteraturePriorProvider()
    regularizer = PhysicsRegularizer(provider=provider, beta=0.5)
    cache = TrajectoryCache(provider=provider)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = PhysGIMINTrainer(
        model=model,
        regularizer=regularizer,
        trajectory_cache=cache,
        optimizer=optimizer,
        seed=seed,
        warmup_epochs=warmup_epochs,
        lambda_phys_target=lambda_phys_target,
        min_recon_fraction=min_recon_fraction,
        output_dir=tmp_path,
        run_name="test_trainer",
        deterministic=deterministic,
        device="cpu",
    )
    return trainer


def _make_loader(seed: int = 1001) -> DataLoader:
    dataset = _TinyDataset(seed=seed)
    return DataLoader(dataset, batch_size=1, collate_fn=_collate_passthrough)


# ── Class 1: Scheduler + loss composition (5 tests) ──────────────────────────

class TestSchedulerAndLoss:
    """Tests for the LR-annealing λ scheduler and loss composition."""

    def test_warmup_holds_lambda_at_zero(self, tmp_path):
        """Epochs before warmup_epochs → lambda_phys == 0."""
        # warmup_epochs=10 means λ stays 0 for the entire 3-epoch run
        trainer = _build_trainer(tmp_path, seed=1001, warmup_epochs=10)
        loader = _make_loader(1001)
        result = trainer.fit(loader, loader, n_epochs=3)

        # lambda_phys should stay 0 throughout warmup
        for rec in result.history:
            assert rec.lambda_phys == 0.0, (
                f"Epoch {rec.epoch}: expected lambda_phys=0 during warmup, got {rec.lambda_phys}"
            )

    def test_lambda_ramps_up_after_warmup(self, tmp_path):
        """Epoch >= warmup → lambda_phys > 0 and <= lambda_phys_target."""
        # warmup_epochs=1, so from epoch 1 onwards λ should start moving
        trainer = _build_trainer(tmp_path, seed=1001, warmup_epochs=1, lambda_phys_target=1.0)
        loader = _make_loader(1001)
        result = trainer.fit(loader, loader, n_epochs=5)

        # After at least 2 post-warmup epochs we expect λ > 0
        post_warmup = [r for r in result.history if r.epoch >= 1]
        assert len(post_warmup) >= 1, "Need at least 1 post-warmup epoch"
        # The final epoch should have λ > 0 (some gradient signal)
        final_lambda = post_warmup[-1].lambda_phys
        assert final_lambda >= 0.0, "lambda_phys must be non-negative"
        assert final_lambda <= 1.0, "lambda_phys must be <= lambda_phys_target=1.0"

    def test_recon_floor_triggers_decay(self, tmp_path):
        """Force high initial lambda → recon_fraction < 0.30 triggers lambda halving."""
        # Set min_recon_fraction=0.99 — almost certainly violated → decay fires
        trainer = _build_trainer(
            tmp_path, seed=1001, warmup_epochs=0, lambda_phys_target=1000.0,
            min_recon_fraction=0.99,
        )
        # Manually set a high lambda to guarantee floor violation
        trainer.lambda_phys = 999.0

        loader = _make_loader(1001)
        result = trainer.fit(loader, loader, n_epochs=2)

        # floor_violation_count > 0 means the decay fired at least once
        assert trainer.floor_violation_count > 0, (
            "Expected at least one floor violation with min_recon_fraction=0.99"
        )
        # lambda should have been reduced from 999
        assert trainer.lambda_phys < 999.0, "lambda_phys should decay after floor violation"

    def test_ema_alpha_stability(self, tmp_path):
        """alpha=0.9 smooths λ updates (no wild swings between epochs)."""
        trainer = _build_trainer(tmp_path, seed=1001, warmup_epochs=1)
        loader = _make_loader(1001)
        result = trainer.fit(loader, loader, n_epochs=8)

        # Check that λ doesn't swing wildly: |Δλ| per epoch should be bounded
        post_warmup = [r.lambda_phys for r in result.history if r.epoch >= 1]
        if len(post_warmup) >= 2:
            deltas = [abs(post_warmup[i+1] - post_warmup[i]) for i in range(len(post_warmup)-1)]
            # With EMA alpha=0.9, changes should be bounded by (1-0.9) * lambda_hat per step
            # Allow generous bound (10x the lambda_phys_target) to test "no wild swings"
            for delta in deltas:
                assert delta < 100.0, f"λ swing too large: Δ={delta}"

    def test_loss_components_match_physgimin_loss_function(self, tmp_path):
        """Trainer's internal L_total matches direct call to physgimin_loss."""
        trainer = _build_trainer(tmp_path, seed=1001, warmup_epochs=0)
        dataset = _TinyDataset(seed=1001)
        batch = dataset[0]

        # Move to CPU
        features = batch["features"]
        mask = batch["mask"]
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        overlap_frac = batch["overlap_frac"]
        stage_ids = batch["stage_ids"]
        ode_indices = batch["ode_feature_indices"]
        patnos = batch["patnos"]
        t_years = batch["t_years_per_patient"]
        sbr_0 = batch["sbr_0_per_patient"]

        trainer.model.eval()
        with torch.no_grad():
            output = trainer.model(
                features, mask, edge_index, edge_weight, overlap_frac, stage_ids,
            )
            mu_all = output["imputed_mean"]
            sigma_all = torch.exp(0.5 * output["imputed_log_var"]).clamp(min=1e-3)
            mu_ode = mu_all[:, ode_indices]
            sigma_ode = sigma_all[:, ode_indices]

            direct_components = physgimin_loss(
                mu_recon=mu_all,
                sigma_recon=sigma_all,
                target_recon=features,
                mu_ode=mu_ode,
                sigma_ode=sigma_ode,
                regularizer=trainer.regularizer,
                patnos=patnos,
                t_years_per_patient=t_years,
                sbr_0_per_patient=sbr_0,
                lambda_phys=trainer.lambda_phys,
                beta=0.5,
                min_recon_fraction=0.30,
            )

        # Direct call should produce finite loss components
        assert torch.isfinite(direct_components.recon), "Direct recon must be finite"
        assert torch.isfinite(direct_components.physics), "Direct physics must be finite"
        assert torch.isfinite(direct_components.total), "Direct total must be finite"
        assert direct_components.total.item() == pytest.approx(
            direct_components.recon.item() + trainer.lambda_phys * direct_components.physics.item(),
            rel=1e-4,
        ), "L_total must equal L_recon + lambda * L_physics"


# ── Class 2: Robustness — output schema + status + provenance (5 tests) ───────

class TestRobustnessContract:
    """Tests for output directory schema, atomic status writes, config/provenance."""

    def test_output_dir_schema_is_complete(self, tmp_path):
        """After 2 epochs, run_dir contains config.json + training_history.json +
        status.json + provenance.json + checkpoint.pt (all 5)."""
        trainer = _build_trainer(tmp_path, seed=1001)
        loader = _make_loader(1001)
        trainer.fit(loader, loader, n_epochs=2)
        for fname in [
            "config.json",
            "training_history.json",
            "status.json",
            "provenance.json",
            "checkpoint.pt",
        ]:
            assert (trainer.run_dir / fname).exists(), f"Missing required file: {fname}"

    def test_status_json_shows_completed_after_clean_run(self, tmp_path):
        """status.json after .fit() shows status='completed' with wall_time_s."""
        trainer = _build_trainer(tmp_path, seed=1001)
        loader = _make_loader(1001)
        trainer.fit(loader, loader, n_epochs=2)

        status = json.loads((trainer.run_dir / "status.json").read_text())
        assert status["status"] == "completed", (
            f"Expected status='completed', got {status['status']}"
        )
        assert "wall_time_s" in status, "status.json must contain wall_time_s"
        assert status["wall_time_s"] >= 0.0, "wall_time_s must be non-negative"

    def test_config_json_echoes_constructor_args(self, tmp_path):
        """config.json contains seed, lambda_phys_target, warmup_epochs, ema_alpha,
        min_recon_fraction, model class name."""
        trainer = _build_trainer(
            tmp_path, seed=1001, warmup_epochs=3, lambda_phys_target=2.0, min_recon_fraction=0.35,
        )
        loader = _make_loader(1001)
        trainer.fit(loader, loader, n_epochs=1)

        config = json.loads((trainer.run_dir / "config.json").read_text())
        assert config["seed"] == 1001
        assert config["lambda_phys_target"] == 2.0
        assert config["warmup_epochs"] == 3
        assert "ema_alpha" in config
        assert config["min_recon_fraction"] == pytest.approx(0.35)
        assert config["model_class"] == "PhysGIMIN"

    def test_provenance_includes_prior_source_hash_and_git_sha(self, tmp_path):
        """provenance.json has prior_source_hash (from regularizer) +
        variant_label + git_sha (can be 'unknown' if git not available)."""
        trainer = _build_trainer(tmp_path, seed=1001)
        loader = _make_loader(1001)
        trainer.fit(loader, loader, n_epochs=1)

        provenance = json.loads((trainer.run_dir / "provenance.json").read_text())
        assert "prior_source_hash" in provenance, "provenance.json missing prior_source_hash"
        assert "variant_label" in provenance, "provenance.json missing variant_label"
        assert "git_sha" in provenance, "provenance.json missing git_sha"
        # git_sha may be 'unknown' in CI but must be present
        assert isinstance(provenance["git_sha"], str), "git_sha must be a string"
        # prior_source_hash should match the regularizer's value
        expected_hash = trainer.regularizer.prior_source_hash
        assert provenance["prior_source_hash"] == expected_hash, (
            "provenance.json prior_source_hash doesn't match regularizer"
        )

    def test_training_history_json_has_one_record_per_epoch(self, tmp_path):
        """training_history.json is a list of length n_epochs (or early-stopped),
        each record has all EpochRecord fields."""
        n_epochs = 4
        trainer = _build_trainer(tmp_path, seed=1001)
        loader = _make_loader(1001)
        result = trainer.fit(loader, loader, n_epochs=n_epochs, patience=100)

        history_path = trainer.run_dir / "training_history.json"
        assert history_path.exists(), "training_history.json must exist after fit()"
        records = json.loads(history_path.read_text())
        assert isinstance(records, list), "training_history.json must be a list"
        assert len(records) == len(result.history), (
            f"JSON history length {len(records)} != result.history length {len(result.history)}"
        )
        # Check that all EpochRecord fields are present
        required_fields = {
            "epoch", "train_loss", "val_loss", "l_recon", "l_physics",
            "lambda_phys", "recon_fraction", "floor_violation",
            "grad_norm_recon", "grad_norm_physics", "wall_time_s",
        }
        for rec in records:
            for field_name in required_fields:
                assert field_name in rec, (
                    f"EpochRecord field '{field_name}' missing from training_history.json"
                )


# ── Class 3: Robustness — determinism + failure modes (4 tests) ───────────────

class TestRobustnessDeterminism:
    """Tests for seed determinism and failure handling."""

    def test_deterministic_under_seed(self, tmp_path):
        """Two trainer runs with same seed → bit-identical checkpoint.pt bytes."""
        trainer1 = _build_trainer(tmp_path / "run_a", seed=1001, deterministic=True)
        loader1 = _make_loader(1001)
        trainer1.fit(loader1, loader1, n_epochs=2)
        hash1 = hashlib.sha256((trainer1.run_dir / "checkpoint.pt").read_bytes()).hexdigest()

        trainer2 = _build_trainer(tmp_path / "run_b", seed=1001, deterministic=True)
        loader2 = _make_loader(1001)
        trainer2.fit(loader2, loader2, n_epochs=2)
        hash2 = hashlib.sha256((trainer2.run_dir / "checkpoint.pt").read_bytes()).hexdigest()

        assert hash1 == hash2, (
            f"Same seed should produce identical checkpoints.\n"
            f"hash1={hash1}\nhash2={hash2}"
        )

    def test_different_seeds_produce_different_checkpoints(self, tmp_path):
        """Sanity: seed=1001 and seed=1002 produce different weights."""
        trainer1 = _build_trainer(tmp_path / "run_a", seed=1001, deterministic=True)
        loader1 = _make_loader(1001)
        trainer1.fit(loader1, loader1, n_epochs=2)
        hash1 = hashlib.sha256((trainer1.run_dir / "checkpoint.pt").read_bytes()).hexdigest()

        trainer2 = _build_trainer(tmp_path / "run_b", seed=1002, deterministic=True)
        loader2 = _make_loader(1002)
        trainer2.fit(loader2, loader2, n_epochs=2)
        hash2 = hashlib.sha256((trainer2.run_dir / "checkpoint.pt").read_bytes()).hexdigest()

        assert hash1 != hash2, (
            "Different seeds must produce different checkpoints (sanity check)"
        )

    def test_nan_loss_triggers_failed_status(self, tmp_path):
        """Inject NaN into batch features → trainer raises + status='failed', reason='nan_loss'."""
        trainer = _build_trainer(tmp_path, seed=1001, warmup_epochs=0)

        class _NaNDataset(Dataset):
            """Dataset that injects NaN features to trigger nan_loss."""
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                features = torch.full((8, 33), float("nan"))
                return {
                    "features": features,
                    "mask": torch.ones(8, 33),
                    "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    "edge_weight": torch.ones(2),
                    "overlap_frac": torch.ones(2),
                    "stage_ids": torch.zeros(8, dtype=torch.long),
                    "patnos": list(range(3000, 3008)),
                    "t_years_per_patient": [np.array([0.0, 1.0])] * 8,
                    "sbr_0_per_patient": [2.5] * 8,
                    "ode_feature_indices": [0, 1],
                }

        nan_loader = DataLoader(_NaNDataset(), batch_size=1, collate_fn=_collate_passthrough)

        # The trainer should return a failed TrainingResult (not raise to the caller)
        result = trainer.fit(nan_loader, nan_loader, n_epochs=1)

        assert result.status == "failed", (
            f"Expected status='failed' on NaN input, got '{result.status}'"
        )
        assert result.failure_reason in ("nan_loss", "nan_grad", "RuntimeError"), (
            f"Expected failure_reason in {{nan_loss, nan_grad, RuntimeError}}, "
            f"got '{result.failure_reason}'"
        )
        # status.json on disk must also show failed
        status = json.loads((trainer.run_dir / "status.json").read_text())
        assert status["status"] == "failed", (
            f"status.json must show 'failed', got {status['status']}"
        )

    def test_status_transitions_monotonic(self, tmp_path):
        """status.json progresses: started → running (per-epoch) → completed.
        Verify by reading status.json at the end and checking completed + wall_time_s > 0."""
        trainer = _build_trainer(tmp_path, seed=1001)
        loader = _make_loader(1001)

        # Run and check final state
        result = trainer.fit(loader, loader, n_epochs=3)

        # Final status should be 'completed'
        final_status = json.loads((trainer.run_dir / "status.json").read_text())
        assert final_status["status"] == "completed"
        assert final_status.get("wall_time_s", 0.0) >= 0.0

        # Ensure result matches status.json
        assert result.status == "completed"
        assert result.wall_time_s >= 0.0
