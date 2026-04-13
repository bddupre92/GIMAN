"""Model-driven digital twin engine backed by trained TemporalGIMAN ensemble.

Replaces the heuristic v1 DataDrivenTwinSimulator with true model-driven
trajectories from the trained TemporalGIMAN (Phase 2). Supports:

1. Baseline trajectory generation via 5-fold ensemble
2. Incremental updates (full reprocessing with audit trail)
3. Ensemble-derived calibrated confidence intervals
4. Temporal counterfactual simulations
5. JSON-serializable state output for persistence

The engine loads all 5 fold checkpoints and runs each forward pass
through the full ensemble, aggregating risk trajectories via mean
with 2.5th/97.5th percentile confidence bands.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..data_processing.longitudinal_assembler import PatientSequence
from ..modeling.temporal_giman import TemporalGIMAN
from .state import CounterfactualSpec
from .temporal_twin_state import TemporalCounterfactualResult, TemporalTwinState
from .twin_registry import TwinRegistry


class TemporalTwinEngine:
    """Model-driven digital twin engine backed by trained TemporalGIMAN ensemble.

    Loads all 5 fold model checkpoints and produces ensemble-aggregated
    risk trajectories with calibrated confidence intervals.

    Args:
        checkpoint_dir: Directory containing model_fold*.pt checkpoints.
        feature_config_path: Path to longitudinal_34.yaml.
        training_config_path: Path to temporal_giman.yaml.
        checkpoint_pattern: Glob pattern for checkpoint files.
        device: Torch device string.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        feature_config_path: Path,
        training_config_path: Path,
        checkpoint_pattern: str = "model_fold*_*.pt",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_pattern = checkpoint_pattern

        # Load feature config for modality dims and feature names
        self.modality_dims, self.feature_names = _load_feature_config(
            feature_config_path
        )

        # Load training config for model architecture params
        self.model_config = _load_training_config(training_config_path)

        # Load ensemble of fold models
        self.checkpoint_paths = sorted(
            self.checkpoint_dir.glob(checkpoint_pattern)
        )
        if not self.checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoints matching '{checkpoint_pattern}' "
                f"in {self.checkpoint_dir}"
            )

        self.models = self._load_ensemble()
        self.n_models = len(self.models)

        # Identify best fold (highest DC-24m) for hidden state selection
        self.best_fold_idx = self._find_best_fold()

        # Compute model version hash
        self.model_version = TwinRegistry.compute_model_version_hash(
            self.checkpoint_paths
        )

        print(
            f"TemporalTwinEngine loaded: {self.n_models} models, "
            f"best_fold={self.best_fold_idx + 1}, "
            f"model_version={self.model_version}"
        )

    def _load_ensemble(self) -> list[TemporalGIMAN]:
        """Load all fold checkpoints into TemporalGIMAN models."""
        models = []
        cfg = self.model_config

        for ckpt_path in self.checkpoint_paths:
            model = TemporalGIMAN(
                modality_dims=self.modality_dims,
                modality_embed_dim=cfg.get("modality_embed_dim", 64),
                cross_modal_heads=cfg.get("cross_modal_heads", 4),
                fused_dim=cfg.get("fused_dim", 128),
                temporal_hidden_dim=cfg.get("temporal_hidden_dim", 64),
                temporal_num_layers=cfg.get("temporal_num_layers", 2),
                gat_hidden_dim=cfg.get("gat_hidden_dim", 64),
                gat_output_dim=cfg.get("gat_output_dim", 64),
                gat_heads=cfg.get("gat_heads", 4),
                gat_layers=cfg.get("gat_layers", 3),
                dropout=cfg.get("dropout", 0.3),
                adaptive_fusion=cfg.get("adaptive_fusion", True),
                observed_threshold=cfg.get("observed_threshold", 0.3),
                graph_k=cfg.get("graph_k", 10),
            ).to(self.device)

            state_dict = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        return models

    def _find_best_fold(self) -> int:
        """Find the best fold index from checkpoint filenames.

        Falls back to fold 0 if the CV summary is not available.
        """
        # Try to load CV summary from the same directory
        summaries = sorted(
            self.checkpoint_dir.glob("cv_summary_*.json"), reverse=True
        )
        if summaries:
            import json

            data = json.loads(summaries[0].read_text(encoding="utf-8"))
            folds = data.get("folds", [])
            if folds:
                best_idx = 0
                best_metric = -1.0
                for i, f in enumerate(folds):
                    dc24 = f.get("final_metrics", {}).get("dc_index_24m", 0.0)
                    if dc24 > best_metric:
                        best_metric = dc24
                        best_idx = i
                return best_idx
        return 0

    def _prepare_input(
        self, sequence: PatientSequence
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Convert a PatientSequence to batched tensors for model input.

        Returns tensors all with batch dimension N=1:
            features: [1, T, F]
            obs_mask: [1, T, F]
            time_months: [1, T]
            seq_mask: [1, T]
            n_visits: [1]
        """
        T = sequence.n_visits
        features = torch.from_numpy(sequence.features).float().unsqueeze(0)
        obs_mask = torch.from_numpy(sequence.obs_mask).float().unsqueeze(0)
        time_months = (
            torch.from_numpy(sequence.time_months).float().unsqueeze(0)
        )
        seq_mask = torch.ones(1, T)
        n_visits = torch.tensor([T], dtype=torch.long)

        return (
            features.to(self.device),
            obs_mask.to(self.device),
            time_months.to(self.device),
            seq_mask.to(self.device),
            n_visits.to(self.device),
        )

    def generate_baseline(
        self, sequence: PatientSequence
    ) -> TemporalTwinState:
        """Generate a baseline digital twin state from a patient's visit history.

        Runs the full ensemble of TemporalGIMAN models and aggregates
        risk trajectories with confidence intervals.

        Args:
            sequence: Patient's longitudinal visit data.

        Returns:
            TemporalTwinState with ensemble-aggregated outputs.
        """
        features, obs_mask, time_months, seq_mask, n_visits = (
            self._prepare_input(sequence)
        )
        T = sequence.n_visits

        ensemble_risks: list[np.ndarray] = []
        ensemble_hazards: list[np.ndarray] = []
        ensemble_scores: list[float] = []
        ensemble_hidden: list[np.ndarray] = []

        for model in self.models:
            with torch.no_grad():
                output = model(
                    features=features,
                    obs_mask=obs_mask,
                    time_months=time_months,
                    seq_mask=seq_mask,
                    n_visits=n_visits,
                )

            risk_traj = output.risk_trajectories[0, :T, 0].cpu().numpy()
            hazard_traj = output.hazard_trajectories[0, :T, 0].cpu().numpy()
            risk_score = output.risk_scores[0, 0].cpu().item()
            hidden = output.hidden_state[:, 0, :].cpu().numpy()

            ensemble_risks.append(risk_traj)
            ensemble_hazards.append(hazard_traj)
            ensemble_scores.append(risk_score)
            ensemble_hidden.append(hidden)

        # Aggregate ensemble
        risk_stack = np.stack(ensemble_risks, axis=0)  # [5, T]
        hazard_stack = np.stack(ensemble_hazards, axis=0)
        score_arr = np.array(ensemble_scores)

        risk_mean = np.mean(risk_stack, axis=0)
        hazard_mean = np.mean(hazard_stack, axis=0)
        score_mean = float(np.mean(score_arr))

        risk_ci_low = np.percentile(risk_stack, 2.5, axis=0)
        risk_ci_high = np.percentile(risk_stack, 97.5, axis=0)
        score_ci_low = float(np.percentile(score_arr, 2.5))
        score_ci_high = float(np.percentile(score_arr, 97.5))

        # Use best-fold hidden state for warm-start
        best_hidden = ensemble_hidden[self.best_fold_idx]

        now = datetime.now(timezone.utc).isoformat()
        data_version = TwinRegistry.compute_data_version_hash(
            sequence.features, sequence.time_months, sequence.patno
        )

        return TemporalTwinState(
            patno=sequence.patno,
            model_version=self.model_version,
            data_version=data_version,
            created_at=now,
            updated_at=now,
            visit_ids=list(sequence.visit_ids),
            time_months=sequence.time_months.tolist(),
            n_visits=sequence.n_visits,
            hidden_state=best_hidden.tolist(),
            risk_trajectory=risk_mean.tolist(),
            hazard_trajectory=hazard_mean.tolist(),
            risk_score=score_mean,
            risk_trajectory_ci_low=risk_ci_low.tolist(),
            risk_trajectory_ci_high=risk_ci_high.tolist(),
            risk_score_ci=[score_ci_low, score_ci_high],
            update_log=[
                {
                    "action": "baseline_generated",
                    "timestamp": now,
                    "n_visits": sequence.n_visits,
                    "risk_score": score_mean,
                    "ensemble_size": self.n_models,
                }
            ],
        )

    def update_twin(
        self,
        twin_state: TemporalTwinState,
        new_features: np.ndarray,
        new_obs_mask: np.ndarray,
        new_time_month: float,
        new_visit_id: str,
        original_sequence: PatientSequence,
    ) -> TemporalTwinState:
        """Update a twin with a new visit via full reprocessing.

        Appends the new visit to the patient's sequence and re-runs
        the full ensemble. The hidden_state in the output can be used
        for future warm-start optimization.

        Args:
            twin_state: Current twin state.
            new_features: [F] feature values for the new visit.
            new_obs_mask: [F] observation mask for the new visit.
            new_time_month: Time in months for the new visit.
            new_visit_id: Visit identifier (e.g., "V08").
            original_sequence: Full patient sequence (for reconstruction).

        Returns:
            Updated TemporalTwinState.
        """
        # Reconstruct updated sequence
        updated_seq = _append_visit(
            original_sequence,
            new_features,
            new_obs_mask,
            new_time_month,
            new_visit_id,
        )

        # Full reprocessing with ensemble
        new_state = self.generate_baseline(updated_seq)

        # Preserve audit trail
        now = datetime.now(timezone.utc).isoformat()
        new_state.created_at = twin_state.created_at
        new_state.update_log = twin_state.update_log + [
            {
                "action": "visit_update",
                "visit_id": new_visit_id,
                "time_month": new_time_month,
                "timestamp": now,
                "prior_risk_score": twin_state.risk_score,
                "new_risk_score": new_state.risk_score,
                "delta_risk": new_state.risk_score - twin_state.risk_score,
            }
        ]

        return new_state

    def simulate_counterfactual(
        self,
        sequence: PatientSequence,
        specs: list[CounterfactualSpec],
    ) -> TemporalCounterfactualResult:
        """Run temporal counterfactual analysis.

        For each intervention spec, perturbs the patient's features
        within the intervention window and re-runs the ensemble.

        Args:
            sequence: Patient's longitudinal visit data.
            specs: List of counterfactual intervention specifications.

        Returns:
            TemporalCounterfactualResult with delta trajectories.
        """
        baseline = self.generate_baseline(sequence)

        cf_trajectories: dict[str, list[float]] = {}
        cf_ci_low: dict[str, list[float]] = {}
        cf_ci_high: dict[str, list[float]] = {}
        delta_risk: dict[str, float] = {}
        delta_trajectory: dict[str, list[float]] = {}

        for spec in specs:
            if spec.feature_name not in self.feature_names:
                continue

            perturbed = self._apply_perturbation(sequence, spec)
            cf_state = self.generate_baseline(perturbed)

            key = f"{spec.feature_name}:{spec.delta:+.3f}"
            cf_trajectories[key] = cf_state.risk_trajectory
            cf_ci_low[key] = cf_state.risk_trajectory_ci_low
            cf_ci_high[key] = cf_state.risk_trajectory_ci_high
            delta_risk[key] = cf_state.risk_score - baseline.risk_score
            delta_trajectory[key] = (
                np.array(cf_state.risk_trajectory)
                - np.array(baseline.risk_trajectory)
            ).tolist()

        return TemporalCounterfactualResult(
            baseline_state=baseline,
            counterfactual_trajectories=cf_trajectories,
            counterfactual_ci_low=cf_ci_low,
            counterfactual_ci_high=cf_ci_high,
            delta_risk=delta_risk,
            delta_trajectory=delta_trajectory,
            intervention_specs=specs,
        )

    def _apply_perturbation(
        self,
        sequence: PatientSequence,
        spec: CounterfactualSpec,
    ) -> PatientSequence:
        """Apply a counterfactual perturbation to a patient's feature matrix.

        Modifies the specified feature within the intervention window
        and returns a new PatientSequence.
        """
        feat_idx = self.feature_names.index(spec.feature_name)
        perturbed_features = sequence.features.copy()

        t_start, t_end = spec.intervention_window
        for t in range(sequence.n_visits):
            if t_start <= sequence.time_months[t] <= t_end:
                perturbed_features[t, feat_idx] += spec.delta
                if spec.bounds is not None:
                    perturbed_features[t, feat_idx] = float(
                        np.clip(
                            perturbed_features[t, feat_idx],
                            spec.bounds[0],
                            spec.bounds[1],
                        )
                    )

        return PatientSequence(
            patno=sequence.patno,
            features=perturbed_features,
            time_months=sequence.time_months.copy(),
            obs_mask=sequence.obs_mask.copy(),
            visit_ids=list(sequence.visit_ids),
            n_visits=sequence.n_visits,
        )

    def generate_cohort_summary(
        self, states: dict[int, TemporalTwinState]
    ) -> dict[str, Any]:
        """Generate summary statistics for a cohort of twin states."""
        if not states:
            return {"n_patients": 0}

        scores = [s.risk_score for s in states.values()]
        return {
            "n_patients": len(states),
            "risk_score_mean": float(np.mean(scores)),
            "risk_score_std": float(np.std(scores)),
            "risk_score_median": float(np.median(scores)),
            "risk_score_q25": float(np.percentile(scores, 25)),
            "risk_score_q75": float(np.percentile(scores, 75)),
            "risk_score_min": float(np.min(scores)),
            "risk_score_max": float(np.max(scores)),
            "model_version": self.model_version,
            "ensemble_size": self.n_models,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_feature_config(
    path: Path,
) -> tuple[dict[str, int], list[str]]:
    """Load modality dims and ordered feature names from YAML config."""
    with open(path) as f:
        config = yaml.safe_load(f)

    modality_dims: dict[str, int] = {}
    feature_names: list[str] = []

    for mod_name, mod_cfg in config["modalities"].items():
        feats = mod_cfg["features"]
        modality_dims[mod_name] = len(feats)
        feature_names.extend(feats)

    return modality_dims, feature_names


def _load_training_config(path: Path) -> dict[str, Any]:
    """Load model architecture parameters from training config YAML."""
    with open(path) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    temporal_cfg = model_cfg.get("temporal", {})
    data_cfg = config.get("data", {})

    return {
        "modality_embed_dim": model_cfg.get("modality_embed_dim", 64),
        "cross_modal_heads": model_cfg.get("cross_modal_heads", 4),
        "fused_dim": model_cfg.get("fused_dim", 128),
        "gat_hidden_dim": model_cfg.get("gat_hidden_dim", 64),
        "gat_output_dim": model_cfg.get("gat_output_dim", 64),
        "gat_heads": model_cfg.get("gat_heads", 4),
        "gat_layers": model_cfg.get("gat_layers", 3),
        "dropout": model_cfg.get("dropout", 0.3),
        "adaptive_fusion": model_cfg.get("adaptive_fusion", True),
        "observed_threshold": model_cfg.get("observed_threshold", 0.3),
        "temporal_hidden_dim": temporal_cfg.get("hidden_dim", 64),
        "temporal_num_layers": temporal_cfg.get("num_layers", 2),
        "graph_k": data_cfg.get("graph_k", 10),
    }


def _append_visit(
    sequence: PatientSequence,
    new_features: np.ndarray,
    new_obs_mask: np.ndarray,
    new_time_month: float,
    new_visit_id: str,
) -> PatientSequence:
    """Create a new PatientSequence with an additional visit appended."""
    features = np.vstack([sequence.features, new_features.reshape(1, -1)])
    obs_mask = np.vstack([sequence.obs_mask, new_obs_mask.reshape(1, -1)])
    time_months = np.append(sequence.time_months, new_time_month)
    visit_ids = list(sequence.visit_ids) + [new_visit_id]

    return PatientSequence(
        patno=sequence.patno,
        features=features,
        time_months=time_months,
        obs_mask=obs_mask,
        visit_ids=visit_ids,
        n_visits=sequence.n_visits + 1,
    )


def truncate_to_n_visits(
    sequence: PatientSequence, n: int
) -> PatientSequence:
    """Create a truncated PatientSequence with only the first n visits."""
    return PatientSequence(
        patno=sequence.patno,
        features=sequence.features[:n].copy(),
        time_months=sequence.time_months[:n].copy(),
        obs_mask=sequence.obs_mask[:n].copy(),
        visit_ids=sequence.visit_ids[:n],
        n_visits=n,
    )
